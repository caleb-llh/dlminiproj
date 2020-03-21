cd ..
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1gxFVai1Ek4JyST2IrKvtA-j68cFiZgFR' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1gxFVai1Ek4JyST2IrKvtA-j68cFiZgFR" -O "pascal-voc-2012.zip" && rm -rf /tmp/cookies.txt
unzip "pascal-voc-2012.zip"
rm "pascal-voc-2012.zip"
cd dlminiproj