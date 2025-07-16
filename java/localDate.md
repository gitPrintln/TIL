## LocalDate 클래스

#### 시각 정보가 생략된 "날짜 정보"를 표현하기 위한 클래스

#### LocalDate 인스턴스는 Immutable 인스턴스

```java
LocalDate today = LocalDate.now();
LocalDate xmas = LocalDate.of(today.getYear(), 12, 25); // 올 해의 크리스마스
LocalDate eve = xmas.minusDays(1); // 올 해의 크리스마스 이브
```