## 필터 스트림

#### 파일 입출력 + 추가적인 기능

`이해` : 파일로부터 int형 데이터 하나를 읽어들이려면,
- 단계 1: 파일로부터 1바이트 4개를 읽어 들인다.
- 단계 2: 읽어 들인 1바이트 데이터 4개를 하나의 int 형 데이터로 조합한다.

###### 아래의 코드는 파일로부터 4바이트 데이터를 읽어들인다.
```java
InputStream in = new FileInputStream("data.dat");
byte buf[] = new byte[4]; // 4바이트 공간 마련
in.read(buf); // 4바이트를 읽어들인다.
```

```java
InputStream in = new FileInputStream("data.dat"); // 입력 스트림 생성
DataInputStream fIn = new DataInputStream(in); // 필터스트림 생성/연결
```

```java
OutputStream out = new FileOutputStream("data.dat"); // 출력 스트림 생성
DataOutputStream fOut = new DataOutputStream(out); // 필터스트림 생성/연결
```

- 필터스트림이 닫히면 연결된 입/출력 스트림도 닫힘
- 