//go:build unit

package service

import (
	"bytes"
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/Wei-Shaw/sub2api/internal/pkg/tlsfingerprint"
	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/require"
)

// geminiScriptedBody 按顺序吐出 chunks，读取遵循上游请求 context：context 取消后返回其错误。
// onChunk 在每个分片完整吐出后调用；hang=true 时吐完不返回 EOF，而是阻塞到 context 取消。
type geminiScriptedBody struct {
	ctx     context.Context
	chunks  []string
	idx     int
	off     int
	onChunk func(idx int)
	hang    bool
}

func (b *geminiScriptedBody) Read(p []byte) (int, error) {
	if err := b.ctx.Err(); err != nil {
		return 0, err
	}
	if b.idx >= len(b.chunks) {
		if b.hang {
			<-b.ctx.Done()
			return 0, b.ctx.Err()
		}
		return 0, io.EOF
	}
	n := copy(p, b.chunks[b.idx][b.off:])
	b.off += n
	if b.off == len(b.chunks[b.idx]) {
		if b.onChunk != nil {
			b.onChunk(b.idx)
		}
		b.idx++
		b.off = 0
	}
	return n, nil
}

func (b *geminiScriptedBody) Close() error { return nil }

// geminiCtxAwareUpstream 模拟真实传输层：上游请求 context 已取消时 Do 直接失败，响应体读取也随之中断。
type geminiCtxAwareUpstream struct {
	status  int
	chunks  []string
	onChunk func(idx int)
	hang    bool
	calls   int
}

func (u *geminiCtxAwareUpstream) Do(req *http.Request, _ string, _ int64, _ int) (*http.Response, error) {
	u.calls++
	if err := req.Context().Err(); err != nil {
		return nil, err
	}
	return &http.Response{
		StatusCode: u.status,
		Header:     http.Header{"Content-Type": []string{"text/event-stream"}},
		Body:       &geminiScriptedBody{ctx: req.Context(), chunks: u.chunks, onChunk: u.onChunk, hang: u.hang},
	}, nil
}

func (u *geminiCtxAwareUpstream) DoWithTLS(req *http.Request, proxyURL string, accountID int64, accountConcurrency int, _ *tlsfingerprint.Profile) (*http.Response, error) {
	return u.Do(req, proxyURL, accountID, accountConcurrency)
}

// newGeminiDrainTestService 返回接入 upstream 的服务；drainSeconds>0 时开启断开续读。
func newGeminiDrainTestService(upstream HTTPUpstream, drainSeconds int) *GeminiMessagesCompatService {
	svc, _ := newGeminiVertexAPIKeyTestService()
	svc.httpUpstream = upstream
	svc.cfg.Gateway.GeminiDisconnectDrainEnabled = drainSeconds > 0
	svc.cfg.Gateway.GeminiDisconnectDrainTimeoutSeconds = drainSeconds
	return svc
}

// newGeminiDrainTestContext 返回请求 context 可取消的 gin.Context，cancel 模拟客户端断开。
func newGeminiDrainTestContext(t *testing.T, w http.ResponseWriter, path string, body []byte) (*gin.Context, context.CancelFunc) {
	t.Helper()
	gin.SetMode(gin.TestMode)
	c, _ := gin.CreateTestContext(w)
	reqCtx, cancel := context.WithCancel(context.Background())
	t.Cleanup(cancel)
	c.Request = httptest.NewRequest(http.MethodPost, path, bytes.NewReader(body)).WithContext(reqCtx)
	return c, cancel
}

func TestNewGeminiDisconnectDrainContext(t *testing.T) {
	t.Run("disabled returns parent", func(t *testing.T) {
		parent, cancel := context.WithCancel(context.Background())
		defer cancel()
		got, release := newGeminiDisconnectDrainContext(parent, 0, 1)
		defer release()
		require.True(t, got == parent)
	})

	t.Run("cancels only after drain timeout once parent is done", func(t *testing.T) {
		parent, cancel := context.WithCancel(context.Background())
		upstreamCtx, release := newGeminiDisconnectDrainContext(parent, 100*time.Millisecond, 1)
		defer release()

		cancel()
		time.Sleep(20 * time.Millisecond)
		require.NoError(t, upstreamCtx.Err(), "客户端断开后上游请求不应立即取消")
		require.Eventually(t, func() bool { return upstreamCtx.Err() != nil }, time.Second, 10*time.Millisecond,
			"超过续读上限后必须取消上游请求，避免无限阻塞")
	})

	t.Run("release cancels and stops pending drain", func(t *testing.T) {
		parent, cancel := context.WithCancel(context.Background())
		upstreamCtx, release := newGeminiDisconnectDrainContext(parent, time.Hour, 1)
		release()
		require.ErrorIs(t, upstreamCtx.Err(), context.Canceled)
		cancel()
		release()
	})
}

func TestGeminiForwardNative_NonStreamClientGoneDrainBillsFullUsage(t *testing.T) {
	reqBody := []byte(`{"contents":[{"role":"user","parts":[{"text":"hi"}]}]}`)
	path := "/v1beta/models/gemini-3-flash-preview:generateContent"

	t.Run("drain enabled", func(t *testing.T) {
		upstream := &geminiCtxAwareUpstream{status: http.StatusOK, chunks: []string{geminiVertexAPIKeyTestResponse}}
		svc := newGeminiDrainTestService(upstream, 30)
		c, cancel := newGeminiDrainTestContext(t, httptest.NewRecorder(), path, reqBody)
		cancel()

		result, err := svc.ForwardNative(c.Request.Context(), c, geminiVertexAPIKeyTestAccount(""), "gemini-3-flash-preview", "generateContent", false, reqBody)
		require.NoError(t, err)
		require.NotNil(t, result)
		require.Equal(t, 10, result.Usage.InputTokens)
		require.Equal(t, 5, result.Usage.OutputTokens)
	})

	t.Run("drain disabled keeps previous behavior", func(t *testing.T) {
		upstream := &geminiCtxAwareUpstream{status: http.StatusOK, chunks: []string{geminiVertexAPIKeyTestResponse}}
		svc := newGeminiDrainTestService(upstream, 0)
		c, cancel := newGeminiDrainTestContext(t, httptest.NewRecorder(), path, reqBody)
		cancel()

		result, err := svc.ForwardNative(c.Request.Context(), c, geminiVertexAPIKeyTestAccount(""), "gemini-3-flash-preview", "generateContent", false, reqBody)
		require.ErrorIs(t, err, context.Canceled)
		require.Nil(t, result)
	})
}

func TestGeminiForwardNative_StreamClientGoneDrainReadsToFinalUsage(t *testing.T) {
	reqBody := []byte(`{"contents":[{"role":"user","parts":[{"text":"hi"}]}]}`)
	path := "/v1beta/models/gemini-3-flash-preview:streamGenerateContent"

	run := func(t *testing.T, drainSeconds int) (*ForwardResult, error) {
		c, cancel := newGeminiDrainTestContext(t, httptest.NewRecorder(), path, reqBody)
		upstream := &geminiCtxAwareUpstream{
			status:  http.StatusOK,
			chunks:  []string{geminiAbortTextChunk, geminiAbortFinalChunk},
			onChunk: func(idx int) { cancel() },
		}
		svc := newGeminiDrainTestService(upstream, drainSeconds)
		return svc.ForwardNative(c.Request.Context(), c, geminiVertexAPIKeyTestAccount(""), "gemini-3-flash-preview", "streamGenerateContent", true, reqBody)
	}

	t.Run("drain enabled bills final usage", func(t *testing.T) {
		result, err := run(t, 30)
		require.NoError(t, err)
		require.True(t, result.ClientDisconnect)
		require.Equal(t, 10, result.Usage.InputTokens)
		require.Equal(t, 5, result.Usage.OutputTokens)
	})

	t.Run("drain disabled stays unbilled without observed usage", func(t *testing.T) {
		result, err := run(t, 0)
		require.ErrorIs(t, err, context.Canceled)
		require.Nil(t, result)
	})
}

func TestGeminiForwardNative_StreamHungUpstreamStopsAtDrainTimeout(t *testing.T) {
	reqBody := []byte(`{"contents":[{"role":"user","parts":[{"text":"hi"}]}]}`)
	c, cancel := newGeminiDrainTestContext(t, httptest.NewRecorder(), "/v1beta/models/gemini-3-flash-preview:streamGenerateContent", reqBody)
	upstream := &geminiCtxAwareUpstream{
		status:  http.StatusOK,
		chunks:  []string{geminiAbortUsageChunk},
		onChunk: func(idx int) { cancel() },
		hang:    true,
	}
	svc := newGeminiDrainTestService(upstream, 1)

	start := time.Now()
	result, err := svc.ForwardNative(c.Request.Context(), c, geminiVertexAPIKeyTestAccount(""), "gemini-3-flash-preview", "streamGenerateContent", true, reqBody)
	elapsed := time.Since(start)

	require.GreaterOrEqual(t, elapsed, 900*time.Millisecond, "断开后应继续等待上游直到续读上限")
	require.Less(t, elapsed, 5*time.Second, "上游挂起时必须在续读上限后退出")
	require.NoError(t, err)
	require.True(t, result.ClientDisconnect)
	require.Equal(t, 2, result.Usage.OutputTokens)
}

func TestGeminiForwardNative_ClientGoneSkipsRetry(t *testing.T) {
	reqBody := []byte(`{"contents":[{"role":"user","parts":[{"text":"hi"}]}]}`)
	upstream := &geminiCtxAwareUpstream{
		status: http.StatusServiceUnavailable,
		chunks: []string{`{"error":{"code":503,"message":"overloaded","status":"UNAVAILABLE"}}`},
	}
	svc := newGeminiDrainTestService(upstream, 30)
	c, cancel := newGeminiDrainTestContext(t, httptest.NewRecorder(), "/v1beta/models/gemini-3-flash-preview:generateContent", reqBody)
	cancel()

	result, err := svc.ForwardNative(c.Request.Context(), c, geminiVertexAPIKeyTestAccount(""), "gemini-3-flash-preview", "generateContent", false, reqBody)
	require.ErrorIs(t, err, context.Canceled)
	require.Nil(t, result)
	require.Equal(t, 1, upstream.calls, "客户端已断开时不应再发起重试")
}

func TestGeminiCompatForward_StreamClientGoneDrainReadsToFinalUsage(t *testing.T) {
	reqBody := []byte(`{"model":"gemini-3-flash-preview","max_tokens":16,"stream":true,"messages":[{"role":"user","content":"hi"}]}`)
	c, cancel := newGeminiDrainTestContext(t, httptest.NewRecorder(), "/v1/messages", reqBody)
	upstream := &geminiCtxAwareUpstream{
		status:  http.StatusOK,
		chunks:  []string{geminiAbortTextChunk, geminiAbortFinalChunk},
		onChunk: func(idx int) { cancel() },
	}
	svc := newGeminiDrainTestService(upstream, 30)

	result, err := svc.Forward(c.Request.Context(), c, geminiVertexAPIKeyTestAccount(""), reqBody)
	require.NoError(t, err)
	require.True(t, result.ClientDisconnect)
	require.Equal(t, 10, result.Usage.InputTokens)
	require.Equal(t, 5, result.Usage.OutputTokens)
}

func TestGeminiForwardAsChatCompletions_WriteFailureDrainReadsToFinalUsage(t *testing.T) {
	reqBody := []byte(`{"model":"gemini-3-flash-preview","stream":true,"messages":[{"role":"user","content":"hi"}]}`)

	run := func(t *testing.T, drainSeconds int) *ForwardResult {
		c, _ := newGeminiDrainTestContext(t, &geminiFailingResponseWriter{header: http.Header{}}, "/v1/chat/completions", reqBody)
		upstream := &geminiCtxAwareUpstream{status: http.StatusOK, chunks: []string{geminiAbortTextChunk, geminiAbortFinalChunk}}
		svc := newGeminiDrainTestService(upstream, drainSeconds)
		result, err := svc.ForwardAsChatCompletions(c.Request.Context(), c, geminiVertexAPIKeyTestAccount(""), reqBody)
		require.NoError(t, err)
		require.True(t, result.ClientDisconnect)
		return result
	}

	t.Run("drain enabled keeps reading after write failure", func(t *testing.T) {
		result := run(t, 30)
		require.Equal(t, 10, result.Usage.InputTokens)
		require.Equal(t, 5, result.Usage.OutputTokens)
	})

	t.Run("drain disabled stops at first write failure", func(t *testing.T) {
		result := run(t, 0)
		require.Zero(t, result.Usage.OutputTokens)
	})
}
