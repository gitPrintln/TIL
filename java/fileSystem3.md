## file System 3

#### 파일 및 디렉토리 생성 및 소멸
```java
Path fp = Paths.get("C:\\Java\\empty.txt");
fp = Files.createFile(fp); // 파일 생성

Path dp1 = Paths.get("C:\\Java\\empty");;
dp1 = Files.createDirectory(dp1); // 디렉토리 생성

Path dp1 = Paths.get("C:\\Java\\2\\empty");;
dp1 = Files.createDirectories(dp1); // 경로 모든 디렉토리 생성
```

#### 파일 간단한 입출력(바이트 단위)
```java
Files.write(fp, buf1, StandardOpenOption.APPEND);
Files.readAllBytes(fp);
```