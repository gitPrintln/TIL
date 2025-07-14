## Stream 중간 연산

#### FlatMap: 1:n 매핑

#### 예제1)

```java
Stream<String> ss1 = Stream.of("MY_AGE", "YOUR_LIFE");

// 아래에서 스트림 생성
Stream<String> ss2 = ss1.flatMap(s -> Arrays.stream(s.split("_")));
ss2.forEach(s -> System.out.print(s + "\t"));
System.out.println();

// 결과
MY AGE YOUR LIFE
```

<br/>

예제2)

```java
ReportCard[] cards = {
    new ReportCard(70, 80, 90),
    new ReportCard(90, 80, 70),
    new ReportCard(80, 90, 80)
}

// ReportCard 인스턴스로 이루어진 스트림 생성
Stream<ReportCard> sr = Arrays.stream(cards);

// 학생들의 점수 정보로 이루어진 스트림 생성
IntStream si = sr.flatMapToInt(
    r -> IntStream.of(r.getKor(), r.getEng(), r.getMath());
)

// 평균을 구하기 위한 최종 연산 average 진행
double avg = si.average().getAsDouble();
System.out.println("avg. " + avg);

// 결과
avg. 80.0
```