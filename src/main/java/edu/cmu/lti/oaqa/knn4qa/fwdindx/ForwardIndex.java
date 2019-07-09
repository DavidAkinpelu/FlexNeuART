/*
 *  Copyright 2014+ Carnegie Mellon University
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */
package edu.cmu.lti.oaqa.knn4qa.fwdindx;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.TreeMap;

import edu.cmu.lti.oaqa.knn4qa.giza.GizaVocabularyReader;
import edu.cmu.lti.oaqa.knn4qa.utils.DataEntryReader;
import edu.cmu.lti.oaqa.knn4qa.utils.Const;

/**
 * 
 * An in-memory forward index. 
 * 
 * <p>A base abstract class for the forward index.
 * The forward index is created from an XML file produced by a pipeline.
 * It collects all unique (space-separated) words that appear in all the fields.
 * These are used to (1) create an in-memory dictionary (2) and a forward index.
 * The forward index can be either fully in-memory index (it is then
 * saved to and loaded from a text file) or the index where documents
 * are stored in binary files. In any case, the dictionary is always
 * stored in memory.</p>
 * 
 * <p><b>NOTE:</b> word IDs start from 1.</p>
 * 
 * <p>
 * In addition, it computes some statistics for each field:
 * </p>
 * <ol> 
 *  <li>An average number of words per document;
 *  <li>The total number of documents;
 *  <li>An number of documents where each word occur;
 * </ol>  
 * 
 * @author Leonid Boytsov
 *
 */
public abstract class ForwardIndex {

  public enum ForwardIndexType {
   lucene,
   memdb,
   inmem,
   unknown // to use as an indicator that a string entry doesn't correspond to the forward index time
  }
  
  public static final ForwardIndexType mIndexTypes[] = {ForwardIndexType.lucene, ForwardIndexType.memdb, ForwardIndexType.inmem};
  public static final WordEntry UNKNOWN_WORD = new WordEntry(-1);
  public static final int MIN_WORD_ID = 1;
  protected static final int PRINT_QTY = 10000;
  
  protected HashMap<String, WordEntry> mStr2WordEntry = new HashMap<String, WordEntry>();
  
  public static ForwardIndexType getIndexType(String type) {
    for (ForwardIndexType itype : mIndexTypes) {
      if (itype.toString().compareToIgnoreCase(type) == 0) {
        return itype;
      }
    }
    return ForwardIndexType.unknown;
  }
  public static String getTypeList() {
   StringBuffer  sb = new StringBuffer();
   for (int i = 0;  i < mIndexTypes.length; ++i) {
     if (i > 0) sb.append(',');
     sb.append(mIndexTypes[i].toString());
   }
   return sb.toString();
  }

  /**
   * Create an index file instance that can be used to create/save index.
   * 
   * @param filePrefix  a prefix of the index file/directories
   * @param indexType a type of the index field
   * @return a  ForwardIndex sub-class instance
   * @throws IOException
   */
  public static ForwardIndex createWriteInstance(String filePrefix, ForwardIndexType indexType) throws IOException {
    return createInstance(filePrefix, indexType);
  }

  public static ForwardIndex createReadInstance(String filePrefix) throws Exception {
    // If for some weird reason more than one index was created, we will try to use the first one
    ForwardIndex res = null;
    
    for (ForwardIndexType itype : mIndexTypes) {
      String indexPrefixFull = getIndexPrefix(filePrefix, itype);  
      File indexDirOrFile = new File(indexPrefixFull);
      
      if (indexDirOrFile.exists()) {
        res = createInstance(filePrefix, itype);
        break;
      }
    }
    
    if (null == res) {
      throw new Exception("No index found at location: " + filePrefix);
    }
    
    res.readIndex();
    return res;
  }
  
  abstract public String[] getAllDocIds();
   
  /**
   * Retrieves a previously stored index.
   * 
   * @param fileName the file generated by the function {@link #save(String)}.
   */
  abstract public void readIndex() throws Exception;
  
  /**
   * Creates an index from one or more files (for a given field name).
   * 
   * @param fieldName         the name of the field (as specified in the SOLR index-file)
   * @param fileNames         an array of files from which the index is created
   * @param bStoreWordIdSeq   if true, we memorize the sequence of word IDs, otherwise only a number of words (doc. len.)
   * @param maxNumRec         the maximum number of records to process
   * @throws Exception 
   */
  public void createIndex(String fieldName, String[] fileNames, 
                         boolean bStoreWordIdSeq,
                         int maxNumRec) throws Exception {    
    mDocQty       = 0;
    mTotalWordQty = 0;
    
    initIndex();
    
    long totalUniqWordQty = 0; // sum the number of uniq words per document (over all documents)
    
    System.out.println("Creating a new in-memory forward index, maximum # of docs to process: " + maxNumRec);
    
    for (String fileName : fileNames) {    
      try (DataEntryReader inp = new DataEntryReader(fileName)) {
 
        Map<String, String>         docFields = null;
        
        for (;mDocQty < maxNumRec && ((docFields = inp.readNext()) != null) ;) {
          
          String docId = docFields.get(Const.TAG_DOCNO);
          
          if (docId == null) {
            System.err.println(String.format("No ID tag '%s', offending DOC #%d", 
                                              Const.TAG_DOCNO, mDocQty));
          }
          
          String text = docFields.get(fieldName);
          if (text == null) text = "";
          if (text.isEmpty()) {
            System.out.println(String.format("Warning: empty field '%s' for document '%s'",
                                             fieldName, docId));
          }
          
          // If the string is empty, the array will contain an emtpy string, but
          // we don't want this
          text=text.trim();
          String words[] = text.isEmpty() ? new String[0] : text.split("\\s+");
 
          // First obtain word IDs for unknown words
          for (int i = 0; i < words.length; ++i) {
            String w = words[i];
            WordEntry wEntry = mStr2WordEntry.get(w);
            if (null == wEntry) {
              wEntry = new WordEntry(MIN_WORD_ID + mStr2WordEntry.size());
              mStr2WordEntry.put(w, wEntry);
            }
          }
          
          DocEntry doc = createDocEntry(words, bStoreWordIdSeq);

          addDocEntry(docId, doc);
          
          HashSet<String> uniqueWords = new HashSet<String>();        
          for (String w: words) uniqueWords.add(w);
          
          // Let's update word co-occurrence statistics
          for (String w: uniqueWords) {
            WordEntry wEntry = mStr2WordEntry.get(w);
            wEntry.mWordFreq++;
          }
          
          ++mDocQty;
          if (mDocQty % PRINT_QTY == 0) {
            System.out.println("Processed " + mDocQty + " documents");
            System.gc();
          }
          mTotalWordQty += words.length;
          totalUniqWordQty += doc.mQtys.length;
        }
        
        postIndexComp();
        
        System.out.println("Finished processing file: " + fileName);
        
        if (mDocQty >= maxNumRec) break;
      }
    }
    
    System.out.println("Final statistics: ");
    System.out.println(
        String.format("Number of documents %d, total number of words %d, average reduction due to keeping only unique words %f",
                      mDocQty, mTotalWordQty, 
                      ((double)mTotalWordQty)/totalUniqWordQty));
  }
  
  public abstract void saveIndex() throws IOException;
  
  protected abstract void sortDocEntries();
  
  /**
   *  Pre-compute some values.
   */
  protected void postIndexComp() {
    // Let's build a list of words & docs sorted by their IDs      
    buildWordListSortedById();
    
    // A mapping from word IDs to word entries.
    // MUST go after buildWordListSortedById()
    buildInt2WordEntry();
    
    sortDocEntries();
    
    mAvgDocLen = mTotalWordQty;
    mAvgDocLen /= mDocQty;
  }

  /**
   * @return an average document length.
   */
  public float getAvgDocLen() {
    return mAvgDocLen;
  }

  /**
   * @return a total number of documents.
   */
  public int getDocQty() {
    return mDocQty;
  }

  /**
   * 
   * @param word
   * @return a WordEntry of a word, or null if the word isn't found.
   */
  public WordEntry getWordEntry(String word) {
    return mStr2WordEntry.get(word);
  }

  /**
   * 
   * @return a WordEntry of a word represented by its ID. If the word
   *         with such ID doesn't exist the null is returned.
   */
  public WordEntry getWordEntry(int wordId) {
    WordEntryExt e = mInt2WordEntryExt.get(wordId);
    
    return e == null ? null : e.mWordEntry;
  }

  public String getWord(int wordId) {
    String res = null;
    
    WordEntryExt e = mInt2WordEntryExt.get(wordId);
    
    if (e != null) {
      return e.mWord;
    }
    
    return res;
  }

  protected void writeHeader(BufferedWriter out) throws IOException {
    // 1. Write meta-info
    out.write(String.format("%d %d", mDocQty, mTotalWordQty));
    out.newLine();
    out.newLine();
    // 2. Write the dictionary
    for (WordEntryExt e: mWordEntSortById) {
      String    w    = e.mWord;
      WordEntry info = e.mWordEntry;
      out.write(String.format("%s\t%d:%d", w, info.mWordId, info.mWordFreq));
      out.newLine();
    }
    out.newLine();
  }

  /**
   * Retrieves an existing document entry.
   * 
   * @param docId document id.
   * @return the document entry of the type {@link DocEntry} or null,
   *         if there is no document with the specified document ID.
   */
  public abstract DocEntry getDocEntry(String docId) throws Exception;

  /**
   * Retrieves an existing document entry and constructs a textual representation.
   * This function needs a positional index.
   * 
   * @param docId document id.
   * @return the document text or null,
   *         if there is no document with the specified document ID.
   */
  public String getDocEntryText(String docId) throws Exception {
    DocEntry e = getDocEntry(docId);
    if (e == null) {
      return null;
    }
    StringBuffer sb = new StringBuffer();
    
    for (int i = 0; i < e.mWordIdSeq.length; ++i) {
      if (i > 0) {
        sb.append(' ');
      }
      int wid = e.mWordIdSeq[i];
      String w = getWord(wid);
      if (w == null) {
        throw new Exception("Looks like bug or inconsistent data, no word for word id: " + wid);
      }
      sb.append(w);
    }
    
    return sb.toString();
  }

  /**
   * Creates a document entry: a sequence of word IDs,
   * plus a list of words (represented again by their IDs)
   * with their frequencies of occurrence in the document.
   * This list is sorted by word IDs. Unknown words
   * have ID -1.
   * 
   * @param words             a list of document words.
   * @param bStoreWordIdSeq   if true, we memorize the sequence of word IDs, otherwise only a number of words (doc. len.)
   * 
   * @return a document entry.
   */
  public DocEntry createDocEntry(String[] words, boolean bStoreWordIdSeq) {
      // TreeMap guarantees that entries are sorted by the wordId
      TreeMap<Integer, Integer> wordQtys = new TreeMap<Integer, Integer>();        
      int [] wordIdSeq = new int[words.length];
      
      for (int i = 0; i < words.length; ++i) {
        String w = words[i];
        WordEntry wEntry = mStr2WordEntry.get(w);
        
        if (wEntry == null) {
          wEntry = UNKNOWN_WORD;
  //        System.out.println(String.format("Warning: unknown token '%s'", w));
        }
          
        int wordId = wEntry.mWordId;
        
        wordIdSeq[i] = wordId;      
        Integer qty = wordQtys.get(wordId);
        if (qty == null) qty = 0;
        ++qty;
        wordQtys.put(wordId, qty);
      }
      
      DocEntry doc = new DocEntry(wordQtys.size(), wordIdSeq, bStoreWordIdSeq);
      
      int k =0;
      
      for (Map.Entry<Integer, Integer> e : wordQtys.entrySet()) {
        doc.mWordIds[k] = e.getKey();
        doc.mQtys[k]    = e.getValue();
        
        k++;
      }
      
      return doc;
    }

  /**
   * Create a table where element with index i, keeps the 
   * probability of the word with ID=i; Thus we can efficiently
   * retrieve probabilities using word IDs.
   * 
   * @param voc     
   *            a GIZA vocabulary from which we take translation probabilities.
   * @return 
   *            a table where element with index i, keeps the 
   *            probability of the word with ID=i
   */
  public float[] createProbTable(GizaVocabularyReader voc) {
    if (mWordEntSortById.length == 0) return new float[0];
    int maxId = mWordEntSortById[mWordEntSortById.length-1].mWordEntry.mWordId;
    float res[] = new float[maxId + 1];
    
    for (WordEntryExt e : mWordEntSortById) {
      int id = e.mWordEntry.mWordId;
      res[id] = (float)voc.getWordProb(e.mWord);
    }
    
    return res;
  }

  void buildInt2WordEntry() {
    for (WordEntryExt e : mWordEntSortById) {
      mInt2WordEntryExt.put(e.mWordEntry.mWordId, e);
    }
  }

  public int getMaxWordId() { return mMaxWordId; }
  
  protected abstract void addDocEntry(String docId, DocEntry doc) throws IOException;

  protected abstract void initIndex() throws IOException;
  
  /**
   * @return an array containing all word IDs
   */
  public int [] getAllWordIds() {
    int [] res = new int [mWordEntSortById.length];
    for (int i = 0; i < mWordEntSortById.length; ++i)
      res[i] = mWordEntSortById[i].mWordEntry.mWordId;
    return res;
  }

  void buildWordListSortedById() {
    mWordEntSortById = new WordEntryExt[mStr2WordEntry.size()];
    
    int k = 0;
    for (Map.Entry<String, WordEntry> e : mStr2WordEntry.entrySet()) {
      mWordEntSortById[k++] = new WordEntryExt(e.getKey(), e.getValue());
      mMaxWordId = Math.max(mMaxWordId, e.getValue().mWordId);
    }
    Arrays.sort(mWordEntSortById);
  }

  HashMap<Integer,WordEntryExt> mInt2WordEntryExt = new HashMap<Integer, WordEntryExt>();
  WordEntryExt[] mWordEntSortById = null;

  protected int mDocQty = 0;
  int mMaxWordId = 0;
  protected long mTotalWordQty = 0;
  float mAvgDocLen = 0;

  /**
   * Read the text-only header (which includes vocabulary info) of the forward file.
   * 
   * @param fileName  input file name (for info purposes only)
   * @param inp  the actual input file opened
   * 
   * @return a line number read plus one
   * 
   * @throws IOException
   * @throws Exception
   */
  protected int readHeader(String fileName, BufferedReader inp) throws IOException, Exception {
    String meta = inp.readLine();

    if (null == meta)
      throw new Exception(
          String.format(
                  "Can't read meta information: the file '%s' may have been truncated.",
                  fileName));

    String parts[] = meta.split("\\s+");
    if (parts.length != 2)
      throw new Exception(
          String.format(
                  "Wrong format, file '%s': meta-information (first line) should contain exactly two space-separated numbers.",
                  fileName));

    try {
      mDocQty = Integer.parseInt(parts[0]);
      mTotalWordQty = Long.parseLong(parts[1]);
    } catch (NumberFormatException e) {
      throw new Exception(String.format(
          "Invalid meta information (should be two-integers), file '%s'.",
          fileName));
    }

    String line = inp.readLine();
    if (line == null || !line.isEmpty()) {
      String.format(
              "Can't read an empty line after meta information: the file '%s' may have been truncated.",
              fileName);
    }

    // First read the dictionary
    int lineNum = 3;
    line = inp.readLine();
    for (; line != null && !line.isEmpty(); line = inp.readLine(), ++lineNum) {
      parts = line.split("\\t");
      if (parts.length != 2) {
        throw new Exception(
            String.format(
                    "Invalid dictionary format (should be two tab-separated parts), line %d, file %s",
                    lineNum, fileName));
      }
      String w = parts[0];
      String[] partSuff = parts[1].split(":");
      if (partSuff.length != 2) {
        throw new Exception(
            String.format(
                    "Invalid dictionary entry format (should end with two colon separated integers), line %d, file %s",
                    lineNum, fileName));
      }

      int wordId = -1;
      int docQty = -1;

      try {
        wordId = Integer.parseInt(partSuff[0]);
        docQty = Integer.parseInt(partSuff[1]);
      } catch (NumberFormatException e) {
        throw new Exception(
            String.format(
                    "Invalid dictionary entry format (an ID or count isn't integer), line %d, file %s",
                    lineNum, fileName));
      }
      if (wordId < MIN_WORD_ID) {
        throw new Exception(
                    String.format("Inconsistent data, wordId %d is too small, should be>= %d", 
                                  wordId, MIN_WORD_ID));
      }
      mStr2WordEntry.put(w, new WordEntry(wordId, docQty));
    }
    if (line == null)
      throw new Exception(
          String.format(
                  "Can't read an empty line (line number %d): the file '%s' may have been truncated.",
                  lineNum, fileName));
    return lineNum;
  }

  private static String getIndexPrefix(String filePrefix, ForwardIndexType indexType) {
    switch (indexType) {
      case lucene: return filePrefix + "." + ForwardIndexType.lucene.toString();       
      case memdb:  return filePrefix + "." + ForwardIndexType.memdb.toString();
      case inmem:  return filePrefix + "." + ForwardIndexType.inmem.toString();
    }
    throw new RuntimeException("Bug: should not reach this point!");
  }
  
  private static ForwardIndex createInstance(String filePrefix, ForwardIndexType indexType) throws IOException {
    String indexPrefixFull = getIndexPrefix(filePrefix, indexType);
    ForwardIndex res = null;
    
    switch (indexType) {
      case lucene:res = new ForwardIndexBinaryLucene(filePrefix, indexPrefixFull); break;
      case memdb: res = new ForwardIndexBinaryMapDb(filePrefix, indexPrefixFull); break;
      case inmem:  res = new InMemForwardIndexText(indexPrefixFull); break;
    }

    return res;    
  }
  
  
}