## I/O stream

#### 프로그램 상당 부분은 입출력과 관련이 있다. 그리고 이들에 대한 자바의 입출력 방식을 가리켜 I/O 모델이라고 한다.

#### ex) 파일, 키보드와 모니터, 그래픽카드와 사운드카드, 프린터와 팩스와 같은 출력장치, 인터넷으로 연결된 서버와 클라이언트

### 입력 스트림(Input Stream) : 실행중인 자바 프로그램으로 데이터를 읽어들이는 스트림, read 읽기,
```java
InputStream in = new FileInputStream("date.dat");
int data = in.read(); // 데이터 읽어들임
```
### 출력 스트림(Output Stream) : 실행중인 자바 프로그램으로부터 데이터를 내보내는 스트림, write 저장
```java
OutputStream out = new FileOutputStream("date.dat");
out.write(7);
```