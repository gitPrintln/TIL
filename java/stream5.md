## stream 최종 연산

#### Collect

```java
[Stream<T>의 메소드]
<R> R collect(Supplier<R> supplier, BiConsumer<R, ? super T> accumulator, BiConsumer<R, R> combiner)

String[] words = {"Hello", "Box", "Robot", "Toy"};
Stream<String> ss = Arrays.stream(words);

List<String> ls = ss.filter(s -> s.length() < 5)
                    .collect(() -> new ArrayList<>(), // 저장소 생성
                    (c, s) -> c.add(s), // 첫 번째 인자 인스턴스 c에 스트림 데이터 s를 담음
                    (lst1, lst2) -> lst1.addAll(lst2)); // 순차스트림 의미 없음

// 그래서 병렬 스트림에서
List<String> ls = ss.parrelel()
                    .filter(s -> s.length() < 5)
                    .collect(() -> new ArrayList<>(), // 저장소 생성
                    (c, s) -> c.add(s), // 첫 번째 인자 인스턴스 c에 스트림 데이터 s를 담음
                    (lst1, lst2) -> lst1.addAll(lst2)); // 순차스트림 의미 없음
```