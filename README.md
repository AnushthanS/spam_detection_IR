# spam_detection_IR
Spam Detection using IR techniques. Designed to be fast, reliable and with all the core Information Retrieval features in mind.

## Project ID 26: Members
- Anushthan Saxena S20210010027
- Mehul Agarwal S20210010142

- This is an implementation of a spam classifier using IR techniques
- All the files required have been zipped along with the code
- run `pip install numpy pandas scikit_learn npz joblib fastapi uvicorn nltk scipy beautifulsoup lxml tqdm` in the terminal within this directory 

- Dataset drive: https://drive.google.com/drive/folders/1wmE--hPk198DysLL-NClNaHyVI8gtszZ?usp=drive_link
- The 'data_extract' python file was used to merge the two datasets used here: 
    trec2007 dataset: https://plg.uwaterloo.ca/~gvcormac/treccorpus07/about.html
    enron dataset corpus: https://www2.aueb.gr/users/ion/data/enron-spam/readme.txt
- Dataset has been merged in the form of a csv file, and there are around 109000 emails with spam/ham classification

- The text preprocessing happens with the appropriately named 'preprocessing_indexing' jupiter notebook
- We dump our vectorizers and tfidf tables to use them without compiling our dataset again and again

## Run:
- backend:
    run the command `uvicorn backend.main:app --reload` in the directory
- frontend:
    in the gui folder:
        run the command `npm i`
        and then `npm run dev` to start the vite server
