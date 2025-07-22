## I/O stream

#### 바이트 단위 입출력 스트림

#### 바이트 단위 파일 복사 프로그램
```java
try(InputStream in = new FileInputStream(src);
    OutputStream out = new FileOutputStream(dst)){
        int data;
        while(true){
            data = in.read(); // 파일로부터 1바이트를 읽는다.
            if(data == -1) // 더 이상 읽어 들일 데이터가 없다면,
                break;
            out.write(data); // 파일에 1바이트를 쓴다.
        }
    }
    catch(IOException e){
        e.printStackTrace();
    }
```

#### 1K 바이트 버퍼 기반 파일 복사 프로그램
```java
try(InputStream in = new FileInputStream(src);
    OutputStream out = new FileOutputStream(dst)){
        byte buf[] = new byte[1024];
        int len;

        while(true){
            data = in.read(buf); // 배열 buf로 데이터를 읽어들이고, (더이상 읽어들일 데이터가 없다면 -1을 반환)
            if(len == -1) // 더 이상 읽어 들일 데이터가 없다면,
                break;
            out.write(buf, 0, len); // len 바이트 만큼 데이터를 저장한다.
        }
    }
    catch(IOException e){
        e.printStackTrace();
    }
```