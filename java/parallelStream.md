## 병렬 스트림(ParallelStream)

- 코어를 나누어서 병렬로 처리
```java
BinaryOperator<String> lc = (s1, s2) -> {
    if(s1.length() > s2.length()){
        return s1;
    }else {
        return s2;
    };
String str = ls.parallelStream() // parallelStream 호출 이후 부터는 병렬 처리 시작
               .reduce("", lc);
```
- 단점: 일을 나누는 것 자체도 일이고 취합하는 연산 자체도 일 -> 리소스를 많이 잡아먹음. -> 시간 재보고 많이 걸릴 거 같으면 병렬스트림 처리