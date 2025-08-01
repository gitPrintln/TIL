## NIO 2 기반의 IO stream 1

#### IO stream -> NIO stream 대체, 생성방법 보완

#### 바이트 스트림 생성(NIO2 기반)

```java
Path fp = Paths.get("data.dat");

try(DataOutputStream out = new DataOutputStream(Files.newOutputStream(fp))) {
    out.write(370);
    out.writeDouble(3.14);
}
```

#### 문자 스트림 생성(NIO2 기반)

```java
Path fp = Paths.get("data.dat");

try(BufferedWriter bw = Files.newBufferedWriter(fp)) {
    bw.write(ks, 0, ks.length());
}
```

- close 같은 그런 부가적인것을 안해도된다.