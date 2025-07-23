## 필터 스트림 2

#### 버퍼링 기능을 제공하는 필터스트림

- 필터스트림의 버퍼에 꽉차야 전송된다.
- 코드 레벨에서 버퍼 스트림하는 것보다 필터스트림으로 하는 것이 훨씬 효율적이고 안정적이다.
- 바이트 단위 복사가 진행되지만 버퍼링되기 때문에 속도가 빠르다.

```java
try(BufferedInputStream in = new BufferInputStream(new FileInputStream(src)); 
    BufferedOutputStream out = new BufferOutputStream(new FileOutputStream(dst))){
        int data;
        while(true){
            data = in.read();
            if(data == -1){
                break;
            }
            out.write(data);
        }
    }catch(IOException e){
        e.printStackTrace();
    }
```

## flush

- 출력 버퍼 스트림에서 사용.
- 어느 정도 차야 전송, 어느 정도 안차도 시간이 지나면 전송 
- 여기서 flush는 버퍼 스트림의 내용을 전송해버림, 출력 버퍼 스트림의 내용을 flush해서 reset 해주는 역할
- 어느 정도 찼을 때 데이터가 날아가면 안될 거 같을 때 전송해야 할 때 전송
- flush를 직접 호출하는 것은 안좋기는 함
- 입력 버퍼 스트림에서는 불필요 -> 의미가 없음