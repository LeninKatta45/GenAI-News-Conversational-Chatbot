curl -X 'POST' \
  'http://0.0.0.0:8080/process_urls/' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "urls": ["https://timesofindia.indiatimes.com/sports/cricket/england-in-india/india-vs-england-kl-rahul-ruled-out-jasprit-bumrah-returns-for-5th-test/articleshow/108102429.cms"
]}'