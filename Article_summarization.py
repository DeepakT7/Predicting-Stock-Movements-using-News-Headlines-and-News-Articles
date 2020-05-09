import nltk
from newspaper import Article
url = input("Enter The Link of the URL below:-")
article = Article(url)
print(123)
article.download()
print(123)
article.parse()
print(123)
article.nlp()
print(123)
article.authors
print(123)
article.publish_date
article.top_image

print('\n \n  \n  \n Summary:-')
a = print(article.summary)
##from newspaper import Article
##
##
##url = "https://www.msn.com/en-in/news/newsindia/political-blame-game-erupts-over-jamia-shooting-incident-heres-who-said-what/ar-BBZtU9v?ocid=spartanntp"
##
###For different language newspaper refer above table
##toi_article = Article(url, language="en") # en for English
##
###To download the article
##toi_article.download()
##
###To parse the article
##toi_article.parse()
##
###To perform natural language processing ie..nlp
##toi_article.nlp()
##
###To extract title
##print("\n \n \n \n \n Article's Title:")
##print(toi_article.title)
##print("n")
##
###To extract text
##print("\n \n \n \n \n Article's Text:")
##print(toi_article.text)
##print("n")
##
###To extract summary
##print("\n \n \n \n \n Article's Summary:")
##print(toi_article.summary)
##print("n")
##
###To extract keywords
##print("\n \n \n \n \n Article's Keywords:")
##print(toi_article.keywords) 
