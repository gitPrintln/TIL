## Spring WebFlux
- Spring WebFlux는 스프링에서 제공하는 비동기(Asynchronous) + 논블로킹(Non-blocking) 기반의 웹 프레임워크

#### 전통적인 Spring MVC
- Tomcat 기반 (Servlet)
- 요청 1개 = 스레드 1개 차지
- 동기 방식 → I/O(DB, API 호출)에서 대기 시간 동안 스레드가 놀고 있음
- 트래픽 증가 시 스레드 고갈로 성능 저하

#### Spring WebFlux
- Netty 기반 또는 Servlet 3 NIO 기반
- 적은 스레드로 많은 요청 처리 가능
- 요청이 들어오면 → 스레드는 작업을 시작하고 → I/O 작업 동안 스레드를 반납
- 나중에 I/O 완료되면 스레드가 다시 처리
→ 리액티브 스트림 기반(Reactor, Mono, Flux)

#### 장점
-  많고 외부 API/DB 호출이 많은 서비스에서 리소스 효율(스레드)이 좋아짐
- 높은 동시성 처리 가능
- 백프레셔(backpressure) 지원 → 데이터 흐름을 제어할 수 있음
- WebClient 사용 가능 (RestTemplate 대체)
#### 단점
- 코드가 더 복잡 (Mono / Flux 개념 필요)
- Debug 어렵고 stacktrace 짧음
- 단순 CRUD 서비스에서는 굳이 사용할 필요 없음

#### WebFlux 사용 -> Webclient 사용하라는 의미

#### 기존 Spring MVC에서는?
- 외부 API 호출 시 보통 RestTemplate을 사용

#### WebFlux에서는?
- 외부 API 호출 시 WebClient를 사용

- WebClient는 비동기 + 논블로킹
```js

WebClient webClient = WebClient.create();

Mono<ResponseDto> responseMono = webClient.get()
.uri(url)
.retrieve()
.bodyToMono(ResponseDto.class);
```
- 이 코드는 스레드를 블로킹하지 않음.
- 응답이 올 때까지 스레드 반납 → 응답 도착 시 다시 처리.
