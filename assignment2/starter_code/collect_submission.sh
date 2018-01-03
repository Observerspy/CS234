rm -f assignment2.zip 
zip -r assignment2.zip . -x "*.pyc" "*.git*" "*weights/*" "*README.md" "*collect_submission.sh" "*events.out*" "*/monitor/*"
