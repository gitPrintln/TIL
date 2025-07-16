## instant 클래스

#### 시각 : 시간의 어느 한 시점
#### 시간 : 어떤 시각에서 어떤 시각까지의 사이

```java
Instant start = Instant.now(); // 현재 시각 정보

Instant end = Instant.now();

Duration between = Duration.between(start, end); // 두 시각의 차이
```

## Duration 클래스
#### 시간의 차이를 계산하는 클래스