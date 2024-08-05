{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Collecting plyvel\n",
      "  Using cached plyvel-1.5.1-cp39-cp39-macosx_11_0_arm64.whl\n",
      "Installing collected packages: plyvel\n",
      "Successfully installed plyvel-1.5.1\n",
      "\n",
      "\u001b[1m[\u001b[0m\u001b[34;49mnotice\u001b[0m\u001b[1;39;49m]\u001b[0m\u001b[39;49m A new release of pip is available: \u001b[0m\u001b[31;49m24.1.2\u001b[0m\u001b[39;49m -> \u001b[0m\u001b[32;49m24.2\u001b[0m\n",
      "\u001b[1m[\u001b[0m\u001b[34;49mnotice\u001b[0m\u001b[1;39;49m]\u001b[0m\u001b[39;49m To update, run: \u001b[0m\u001b[32;49mpip install --upgrade pip\u001b[0m\n"
     ]
    }
   ],
   "source": [
    "\n",
    "!export LEVELDB_INCLUDE=/usr/local/include\n",
    "!export LEVELDB_LIB=/usr/local/lib\n",
    "!pip install plyvel"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [
    {
     "ename": "AttributeError",
     "evalue": "module 'leveldb' has no attribute 'LevelDB'",
     "output_type": "error",
     "traceback": [
      "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[0;31mAttributeError\u001b[0m                            Traceback (most recent call last)",
      "Cell \u001b[0;32mIn[4], line 4\u001b[0m\n\u001b[1;32m      1\u001b[0m \u001b[38;5;28;01mimport\u001b[39;00m \u001b[38;5;21;01mleveldb\u001b[39;00m\n\u001b[1;32m      3\u001b[0m \u001b[38;5;66;03m# Open the LevelDB database\u001b[39;00m\n\u001b[0;32m----> 4\u001b[0m db \u001b[38;5;241m=\u001b[39m \u001b[43mleveldb\u001b[49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43mLevelDB\u001b[49m(\u001b[38;5;124m'\u001b[39m\u001b[38;5;124m/Users/pratham/Desktop/sentencepiece/build/src/sentence_db\u001b[39m\u001b[38;5;124m'\u001b[39m)\n\u001b[1;32m      6\u001b[0m \u001b[38;5;66;03m# Iterate through key-value pairs\u001b[39;00m\n\u001b[1;32m      7\u001b[0m \u001b[38;5;28;01mfor\u001b[39;00m key, value \u001b[38;5;129;01min\u001b[39;00m db\u001b[38;5;241m.\u001b[39mRangeIter():\n",
      "\u001b[0;31mAttributeError\u001b[0m: module 'leveldb' has no attribute 'LevelDB'"
     ]
    }
   ],
   "source": [
    "import leveldb\n",
    "\n",
    "# Open the LevelDB database\n",
    "db = leveldb.LevelDB('/Users/pratham/Desktop/sentencepiece/build/src/sentence_db')\n",
    "\n",
    "# Iterate through key-value pairs\n",
    "for key, value in db.RangeIter():\n",
    "    print(f'Key: {key.decode(\"utf-8\")}, Value: {value.decode(\"utf-8\")}')"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "sp",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.19"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
