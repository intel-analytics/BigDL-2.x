mkdir password && cd password
openssl genrsa -out key.txt 2048
echo "YOUR_PASSWORD" | openssl rsautl -inkey key.txt -encrypt >output.bin
