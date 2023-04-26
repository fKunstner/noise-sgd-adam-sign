
dir=$1
cd "$dir"

mkdir -p wikitext-2

echo "- Downloading WikiText-2 (WT2)"
wget --no-check-certificate --quiet --continue https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip
unzip -q wikitext-2-v1.zip
cd wikitext-2
mv wiki.train.tokens train.txt
mv wiki.valid.tokens valid.txt
mv wiki.test.tokens test.txt
cd ..