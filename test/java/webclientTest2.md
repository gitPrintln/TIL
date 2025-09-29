## “외부 API를 조회해서 데이터를 받아오는 서비스”

#### Mock 기반 단위 테스트 + Native(실제 Stub 서버) 테스트

###### 1. 예시 시나리오

외부 API : GET /users/{id} → UserResponse JSON 반환

우리 서비스 : UserService.getUser(id)

테스트 : Mock으로 외부 호출 결과 주입 / Stub 서버로 실제 호출

#### (1) 외부 API 클라이언트

```java
@Component
public class ExternalUserClient {

    private final WebClient webClient;

    public ExternalUserClient(WebClient.Builder builder) {
        this.webClient = builder.baseUrl("https://api.example.com").build();
    }

    public UserResponse getUser(Long id) {
        return webClient.get()
                .uri("/users/{id}", id)
                .retrieve()
                .bodyToMono(UserResponse.class)
                .block();
    }
}

```

#### (2) 서비스

```java
@Service
public class UserService {

    private final ExternalUserClient externalUserClient;

    public UserService(ExternalUserClient externalUserClient) {
        this.externalUserClient = externalUserClient;
    }

    public UserDto getUser(Long id) {
        UserResponse res = externalUserClient.getUser(id);
        return new UserDto(res.getName(), res.getEmail());
    }
}

```

#### 2. Mock 기반 단위 테스트
```java
@SpringBootTest
class UserServiceMockTest {

    @Autowired
    private UserService userService;

    @MockBean
    private ExternalUserClient externalUserClient; // WebClient 대신 Mock

    @Test
    void testGetUser_withMock() {
        // given
        UserResponse mockRes = new UserResponse("Alice", "alice@test.com");
        when(externalUserClient.getUser(1L)).thenReturn(mockRes);

        // when
        UserDto dto = userService.getUser(1L);

        // then
        assertAll(
            () -> assertEquals("Alice", dto.getName()),
            () -> assertEquals("alice@test.com", dto.getEmail())
        );

        // 외부 호출이 1번 일어났는지 확인
        verify(externalUserClient, times(1)).getUser(1L);
    }
}

```

#### 3. Native(Stub 서버) 테스트 (예: WireMock)
```java
@SpringBootTest
@AutoConfigureWireMock(port = 0) // 랜덤 포트 Stub 서버 띄움
class UserServiceNativeTest {

    @Autowired
    private UserService userService;

    @Test
    void testGetUser_withStubServer() {
        // given: WireMock으로 Stub 응답 정의
        stubFor(get(urlEqualTo("/users/1"))
            .willReturn(aResponse()
                .withHeader("Content-Type", "application/json")
                .withBody("""
                    {"name":"Bob","email":"bob@test.com"}
                """)));

        // when
        UserDto dto = userService.getUser(1L);

        // then
        assertAll(
            () -> assertEquals("Bob", dto.getName()),
            () -> assertEquals("bob@test.com", dto.getEmail())
        );
    }
}

```