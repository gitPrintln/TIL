## WebFlux 전체 아키텍처 구조 (한눈에 보기)

```csharp
          ┌────────────────────────────┐
          │        클라이언트          │
          └───────────────┬────────────┘
                          요청
                          ▼
                [Reactive Server Adapter]
         ┌─────────────────┴─────────────────┐
         │  Netty   │ Undertow │ Jetty │ Tomcat │
         └─────────────────┬─────────────────┘
                         이벤트 루프
                          ▼
               [HttpHandler(WebFlux Core)]
                          ▼
        ┌─────────────────────────────────────┐
        │    DispatcherHandler (중앙 라우팅)  │
        └─────────────────────────────────────┘
   ┌──────────────┬───────────────┬──────────────┐
   │ HandlerMapping │ HandlerAdapter │ ResultHandler │
   └──────────────┴───────────────┴──────────────┘
                          ▼
                     Controller / RouterFunction / Handler
                          ▼
                       Mono / Flux 처리(Reactor)
                          ▼
            [Reactive Streams: Publisher → Subscriber]
                          ▼
                    응답 생성, 버퍼링, 전송
                          ▼
                   Netty 이벤트 루프에서 응답

```

#### WebFlux의 핵심 구조 요약
```js
Server Layer (I/O 처리)
WebFlux Core Layer (HttpHandler → DispatcherHandler)
Application Layer (Controller, HandlerFunction)
```

#### Server Layer: Reactive Server Adapter
- Netty (권장)

- Undertow

- Jetty

- Tomcat (NIO 모드)

- HTTP 요청을 읽고
- 논블로킹 방식으로 내부에 전달하고
- 이벤트 루프 스레드로 작업을 수행하고
- HttpHandler의 결과를 다시 응답으로 반환

#### WebFlux Core Layer (핵심)
- Server → HttpHandler → DispatcherHandler → HandlerMapping