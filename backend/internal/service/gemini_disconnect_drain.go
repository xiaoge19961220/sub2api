package service

import (
	"context"
	"sync"
	"time"

	"github.com/Wei-Shaw/sub2api/internal/pkg/logger"
	"github.com/gin-gonic/gin"
	"go.uber.org/zap"
)

// geminiDisconnectDrainTimeout 返回客户端断开后继续读取上游的最长时间；未开启时返回 0。
func (s *GeminiMessagesCompatService) geminiDisconnectDrainTimeout() time.Duration {
	if s == nil || s.cfg == nil || !s.cfg.Gateway.GeminiDisconnectDrainEnabled {
		return 0
	}
	return time.Duration(s.cfg.Gateway.GeminiDisconnectDrainTimeoutSeconds) * time.Second
}

// newGeminiDisconnectDrainContext 返回发往上游的请求 context。drainTimeout<=0 时原样返回 parent。
// 否则返回保留 parent 取值、但不随 parent 立即取消的 context：parent 结束（客户端断开）后
// 再等 drainTimeout 才取消，让已受理的上游生成读完以拿到完整用量，同时避免上游挂起时无限阻塞。
// 调用方必须在响应体读完后调用 release。
func newGeminiDisconnectDrainContext(parent context.Context, drainTimeout time.Duration, accountID int64) (context.Context, func()) {
	if parent == nil || drainTimeout <= 0 {
		return parent, func() {}
	}
	upstreamCtx, cancel := context.WithCancel(context.WithoutCancel(parent))
	var mu sync.Mutex
	var timer *time.Timer
	released := false
	stopAfter := context.AfterFunc(parent, func() {
		mu.Lock()
		defer mu.Unlock()
		if released {
			return
		}
		timer = time.AfterFunc(drainTimeout, cancel)
		logger.FromContext(parent).Info("gemini.disconnect_drain_started",
			zap.Int64("account_id", accountID),
			zap.Int64("drain_timeout_ms", drainTimeout.Milliseconds()),
		)
	})
	var once sync.Once
	return upstreamCtx, func() {
		once.Do(func() {
			_ = stopAfter()
			mu.Lock()
			released = true
			if timer != nil {
				timer.Stop()
			}
			mu.Unlock()
			cancel()
		})
	}
}

// geminiClientGone 报告下游客户端是否已断开。
func geminiClientGone(c *gin.Context) bool {
	return c != nil && c.Request != nil && c.Request.Context().Err() != nil
}
