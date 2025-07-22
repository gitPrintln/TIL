## I/O stream

#### 파일 입출력 스트림 생성 예시

```java
OutputStream out = new FileOutputStream("data.dat");
out.write(7); // 7을 저장
out.close(); // 출력 스트림 종료
```

```java
InputStream in = new FileInputStream("data.dat");
int dat = in.read(); // 데이터 읽음
in.close(); // 입력 스트림 종료
```