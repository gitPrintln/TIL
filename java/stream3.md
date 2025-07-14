## Stream 중간 연산

#### 정렬(sorted 메서드)

Stream<T> sorted(); <br/>
Stream<T> sorted(Comparator<? super T> comparator) <br/>
IntStream<T> sorted(); <br/>
LongStream<T> ... <br/>
DoubleStream<T> ...

#### 루핑(Looping)
-> 대표적인 루핑 연산(forEach)는 최종연산이지만
'중간 연산'으로 peek이 있다.
Stream<T> peek(Consumer<? super T> action) <br/>
IntStream<T> peek(IntConsumer action); <br/>
LongStream<T> ... <br/>
DoubleStream<T> ... 

```java
IntStream.of(1, 3, 5).peek(d -> System.out.print(d + "\t"));

IntStream.of(5, 3, 1).peek(d -> System.out.print(d + "\t")).sum();
```