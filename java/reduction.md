## 리덕션 : 데이터를 축소하는 연산

- reduce 메서드 기반의 연산
```java
T reduce(T identity, BinaryOperator<T> accumulator) // stream<T>에 존재
BinaryOperator<T> -> T apply(T t1, T t2)

ex) 
List ls = Arrays.asList("Box", "Simple", "Complex", "Robot");
BinaryOperator<String> lc = (s1, s2) -> {
    if(s1.length() > s2.length()){
        return s1;
    }else {
        return s2;
    };
String str = ls.stream().reduce("", lc); // 스트림이 빈 경우 "" 반환
// 주의 개념: reduce에 첫 번째 인자도 같이 경쟁됨.(7글자의 string이 들어갈 경우 첫 번째 인자가 반환)
}
```