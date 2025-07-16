## LocalTime 클래스

#### 시각 정보를 나타내는 클래스
#### 시, 분, 초를 표현
#### 마찬가지로 시간의 차이를 계산 : Duration 클래스

```java
LocalTime now = LocalTime.now(); // 현재 시각

LocalTime mt = now.plusHours(2); // 2시간 뒤
mt = mt.plusMinutes(10); // 10분 뒤

LocalTime start = LocalTime.of(14, 24, 35); // 14:24:35
LocalTime end = LocalTime.of(17, 31, 19); // 17:31:19
Duration between = Duration.between(start, end); // PT3H6M44S
```

## LocalDateTime : 두 날짜, 두 시간의 차이를 나타냄

```java
// 현재 날짜와 시각
LocalDateTime dt = LocalDateTime.now();

// 22시간 35분 뒤
LocalDateTime mt = dt.plusHours(22);
mt = mt.plusMinutes(35);

// 출력하면 : 2025-07-16T22:19:12.258 -> 2025-07-17T20:54:12.258
// 시각의 차 : Period 클래스의 between 메서드 : toLocalDate() 호출
// 날짜의 차 : Duration 클래스의 between 메서드 : toLocalTime() 호출 
```