## file System 2

#### paths 와 path 클래스

- java.nio.file.path
- 파일 및 디렉토리 경로 표현을 위해 자바 7 에서 추가된 인터페이스
```java
Path path = paths.get("C:\\javastudy\\PathDemo.java");
```

```java
Path getRoot() // 루트 디렉토리 반환
Path getParent() // 부모 디렉토리 반환
Path getFileName() // 파일 이름 반환
```

#### 현재 디렉토리 정보 출력 
```java
Path cur = Paths.get("");
String cdir;

if(cur.isAbsolute())
    cdir = cur.toString();
else 
    cdir = cur.toAbsolutePath().toString();
    
```