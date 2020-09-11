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
package edu.cmu.lti.oaqa.flexneuart.fwdindx;


import java.io.File;
import java.io.IOException;
import java.nio.file.Paths;

import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.core.WhitespaceAnalyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.StoredField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.IndexWriterConfig.OpenMode;
import org.apache.lucene.queryparser.classic.QueryParser;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.util.BytesRef;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import edu.cmu.lti.oaqa.flexneuart.cand_providers.LuceneCandidateProvider;
import edu.cmu.lti.oaqa.flexneuart.utils.Const;

/**
 * @author Leonid Boytsov
 *
 */
public class ForwardIndexBinaryLucene extends ForwardIndexBinaryBase {
  
  private static final Logger logger = LoggerFactory.getLogger(ForwardIndexBinaryLucene.class);
  public static final int COMMIT_INTERV = 2000000;
  
  protected String mBinDir;

  public ForwardIndexBinaryLucene(String vocabAndDocIdsFile, String binDir) throws IOException {
    super(vocabAndDocIdsFile);
    mBinDir = binDir;
  }
  
  @Override
  protected void initIndex() throws IOException {
    mDocIds.clear();
   
    File outputDir = new File(mBinDir);
    if (!outputDir.exists()) {
      if (!outputDir.mkdirs()) {
        logger.error("couldn't create " + outputDir.getAbsolutePath());
        System.exit(1);
      }
    }
    if (!outputDir.isDirectory()) {
      logger.error(outputDir.getAbsolutePath() + " is not a directory!");
      System.exit(1);
    }
    if (!outputDir.canWrite()) {
      logger.error("Can't write to " + outputDir.getAbsolutePath());
      System.exit(1);
    }
    Analyzer analyzer = new WhitespaceAnalyzer();
    FSDirectory       indexDir    = FSDirectory.open(Paths.get(mBinDir));
    IndexWriterConfig indexConf   = new IndexWriterConfig(analyzer);
    
    /*
    OpenMode.CREATE creates a new index or overwrites an existing one.
    https://lucene.apache.org/core/6_0_0/core/org/apache/lucene/index/IndexWriterConfig.OpenMode.html#CREATE
    */
    indexConf.setOpenMode(OpenMode.CREATE); 
    indexConf.setRAMBufferSizeMB(LuceneCandidateProvider.DEFAULT_RAM_BUFFER_SIZE);
    
    indexConf.setOpenMode(OpenMode.CREATE);
    mIndexWriter = new IndexWriter(indexDir, indexConf);  
  }
  
  private String getDocEntryStr(String docId) throws Exception {
    QueryParser parser = new QueryParser(Const.TAG_DOCNO, mAnalyzer);
    Query       queryParsed = parser.parse(docId);
    
    TopDocs     hits = mSearcher.search(queryParsed, 1);
    ScoreDoc[]  scoreDocs = hits.scoreDocs;
    if (scoreDocs != null && scoreDocs.length == 1) {
      Document doc = mSearcher.doc(scoreDocs[0].doc);
      return doc.get(Const.TAG_DOC_ENTRY);
    }
    return null;
  }
  
  private BytesRef getDocEntryBin(String docId) throws Exception {
    QueryParser parser = new QueryParser(Const.TAG_DOCNO, mAnalyzer);
    Query       queryParsed = parser.parse(docId);
    
    TopDocs     hits = mSearcher.search(queryParsed, 1);
    ScoreDoc[]  scoreDocs = hits.scoreDocs;
    if (scoreDocs != null && scoreDocs.length == 1) {
      Document doc = mSearcher.doc(scoreDocs[0].doc);
      return doc.getBinaryValue(Const.TAG_DOC_ENTRY);
    }
    return null;
  }

  @Override
  public String getDocEntryRaw(String docId) throws Exception {
    if (!mIsRaw) {
      throw new Exception("An attempt to read the raw entry for the parsed field, directory: " + mBinDir);
    }
    return getDocEntryStr(docId);
  }

  @Override
  public DocEntryParsed getDocEntryParsed(String docId) throws Exception {
    if (mIsRaw) {
      throw new Exception("An attempt to read the parsed entry for the raw field, directory: " + mBinDir);
    }
    BytesRef docBin = getDocEntryBin(docId);
    if (docBin != null) {
      return DocEntryParsed.fromBinary(docBin.bytes);
    }
    return null;
  }
  
  @Override
  public void readIndex() throws Exception {
    readHeaderAndDocIds();
    
    mReader = DirectoryReader.open(FSDirectory.open(Paths.get(mBinDir)));
    mSearcher = new IndexSearcher(mReader);
    
    logger.info("Finished loading context from dir: " + mBinDir);
  }
  
  @Override
  protected void addDocEntryRaw(String docId, String docText) throws IOException {   
    
    Document luceneDoc = new Document();
    luceneDoc.add(new StringField(Const.TAG_DOCNO, docId, Field.Store.YES));
    luceneDoc.add(new StoredField(Const.TAG_DOC_ENTRY, docText));
    
  	mDocIds.add(docId);
  
    if (mDocIds.size() % COMMIT_INTERV == 0) {
      logger.info("Committing");
      mIndexWriter.commit();
    }
    
    // Index writers should be completely thread-safe 
    mIndexWriter.addDocument(luceneDoc);
    
  }
  
  @Override
  protected void addDocEntryParsed(String docId, DocEntryParsed doc) throws IOException {

    Document luceneDoc = new Document();
    luceneDoc.add(new StringField(Const.TAG_DOCNO, docId, Field.Store.YES));
    luceneDoc.add(new StoredField(Const.TAG_DOC_ENTRY, new BytesRef(doc.toBinary())));
    
    mDocIds.add(docId);
  
    if (mDocIds.size() % COMMIT_INTERV == 0) {
      logger.info("Committing");
      mIndexWriter.commit();
    }
    
    // Index writers should be completely thread-safe 
    mIndexWriter.addDocument(luceneDoc);
  }

  @Override
  public void saveIndex() throws IOException {
    writeHeaderAndDocIds();
   
    mIndexWriter.commit();
    mIndexWriter.close();
  }

  private IndexWriter mIndexWriter;
  private DirectoryReader mReader;
  private IndexSearcher mSearcher;
  private Analyzer      mAnalyzer = new WhitespaceAnalyzer();


}
