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