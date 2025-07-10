## 스트림(Stream)

#### 예제1)

```java
int[] ar = {1, 2, 3, 4, 5};
IntStream stm1 = Arrays.stream(ar); // 배열 ar로부터 스트림 생성
IntStream stm2 = stm1.filter(n -> n%2 == 1); // 중간 연산 진행
int sum = stm2.sum(); // 최종 연산 진행
System.out.print(sum);

---
int[] ar = {1, 2, 3, 4, 5};
int sum = Arrays.stream(ar) // 스트림 생성
          .filter(n -> n%2 == 1) // filter 통과
          .sum(); // sum 통과 결과 반환
System.out.print(sum);
```
<br/>

`특징`
- 스트림 생성: 배열로 생성, 컬렉션 인스턴스를 대상으로 생성, of 메서드로 직접 데이터 전달
- IntStream, DoubleStream, LongStream 등등 다양한 of메서드들이 있음
- 기존의 stream을 .parallel() 호출 이후 -> 이후 연산은 병렬 처리됨
- 스트림끼리 연결:
```java
Stream.concat(ss1,ss2)
      .forEach(s -> System.out.println(S));
```