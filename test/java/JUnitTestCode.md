## A(Arrange-Act-Assert) 패턴과 Given-When-Then 패턴

#### 테스트를 구조화해서 가독성을 높이고, 협업 시 의도를 명확히 드러내는 데 

ex1)
1. AAA 패턴 (Arrange - Act - Assert)
테스트의 3단계 패턴입니다.

Arrange (준비)
테스트 실행에 필요한 객체, 데이터, Mock 등을 준비하는 단계
예: User user = new User("test", "pw");

Act (실행)
실제 테스트할 메서드나 로직을 실행하는 단계
예: boolean result = userService.login(user);

Assert (검증)
기대한 값과 실제 결과가 같은지 검증하는 단계
예: assertTrue(result);

```java
@Test
void loginSuccess_AAA() {
    // Arrange
    User user = new User("test", "pw");
    UserService userService = new UserService();

    // Act
    boolean result = userService.login(user);

    // Assert
    assertTrue(result);
}
```
ex2)
2. Given-When-Then 패턴
BDD(Behavior Driven Development, 행위 주도 개발)에서 차용한 표현 방식인데, 사실 AAA와 거의 동일합니다.
차이는 비즈니스 시나리오처럼 읽히도록 표현한다는 점이에요.

Given (주어진 상황)
테스트를 하기 위한 초기 조건, 환경, 데이터 준비
예: "유저가 올바른 아이디/비밀번호를 가지고 있다"

When (실행 조건, 행동)
특정 동작을 실행했을 때
예: "로그인을 시도했을 때"

Then (결과 검증)
기대되는 결과
예: "로그인에 성공해야 한다"

```java
@Test
void loginSuccess_GivenWhenThen() {
    // Given: 올바른 사용자 정보가 주어졌을 때
    User user = new User("test", "pw");
    UserService userService = new UserService();

    // When: 로그인을 시도하면
    boolean result = userService.login(user);

    // Then: 로그인이 성공해야 한다
    assertTrue(result);
}
```