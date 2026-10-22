module github.com/beingPro007/lk-go-agent-sdk/plugins/deepgram

go 1.26

toolchain go1.26.5

require (
	github.com/beingPro007/lk-go-agent-sdk v0.0.0
	github.com/gorilla/websocket v1.5.4-0.20250319132907-e064f32e3674
)

require golang.org/x/net v0.55.0 // indirect

replace github.com/beingPro007/lk-go-agent-sdk => ../..
