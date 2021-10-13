package edu.cmu.lti.oaqa.flexneuart.simil_func;

import edu.cmu.lti.oaqa.flexneuart.fwdindx.DocEntryParsed;
import edu.cmu.lti.oaqa.flexneuart.fwdindx.ForwardIndex;
import edu.cmu.lti.oaqa.flexneuart.fwdindx.WordEntry;

/**
 * Just the cosine similarity between TF*IDF vectors (using BM25 IDF).
 * 
 * @author Leonid Boytsov
 *
 */
public class CosineTextSimilarity extends TFIDFSimilarity {
  public CosineTextSimilarity(ForwardIndex fieldIndex) {
    mFieldIndex = fieldIndex;
  }
  
  @Override
  protected float computeIDF(float docQty, WordEntry e) {
    float n = e.mWordFreq;
    return (float)Math.log(1 + (docQty - n + 0.5D)/(n + 0.5D));
  }

  final ForwardIndex mFieldIndex;
  
  
  /**
   * Computes the similarity between the query (represented by
   * a DocEntry object) and the document (also represented by a DocEntry object)
   * 
   * @param query
   * @param document
   * @return
   */
  @Override
  public float compute(DocEntryParsed query, DocEntryParsed doc) {
    float score = 0;
    
    int   queryTermQty = query.mWordIds.length;

    float normQuery =0;    
    for (int iQuery = 0; iQuery < queryTermQty; ++iQuery) {
      final int queryWordId = query.mWordIds[iQuery];
      if (queryWordId >= 0) {
        float idf = getIDF(mFieldIndex, queryWordId);
        float w = query.mQtys[iQuery]*idf;
        normQuery += w * w; 
      }
    }
    
    int   docTermQty = doc.mWordIds.length;
    
    float normDoc = 0;
    for (int iDoc = 0; iDoc < docTermQty; ++iDoc) {
      final int docWordId   = doc.mWordIds[iDoc];
      // docWordId >= 0 should always be non-negative (unlike queryWordId, which can be -1 for OOV words 
      float idf = getIDF(mFieldIndex, docWordId);
      float w = doc.mQtys[iDoc]*idf;
      normDoc += w * w;
    }
    
    int   iQuery = 0, iDoc = 0;
    
    while (iQuery < queryTermQty && iDoc < docTermQty) {
      final int queryWordId = query.mWordIds[iQuery];
      final int docWordId   = doc.mWordIds[iDoc];
      
      if (queryWordId < docWordId) ++iQuery;
      else if (queryWordId > docWordId) ++iDoc;
      else { 
        // Here queryWordId == docWordId
        float idf = getIDF(mFieldIndex, docWordId);
        score +=  query.mQtys[iQuery] * idf * doc.mQtys[iDoc] * idf;
        
        ++iQuery; ++iDoc;
      }
    }
    
    return score /= Math.sqrt(Math.max(1e-6, normQuery * normDoc));
  }
  
  public TrulySparseVector getSparseVector(DocEntryParsed e, boolean isQuery) {
    int qty = 0;
    for (int wid : e.mWordIds)
      if (wid >= 0) qty++;
    TrulySparseVector res = new TrulySparseVector(qty);

    float norm = 0;
    // Getting vector values
    for (int i = 0, id=0; i < e.mWordIds.length; ++i) {
      int wordId = e.mWordIds[i];
      if (wordId < 0) continue;
      float IDF = getIDF(mFieldIndex, wordId);
      float tf = e.mQtys[i];
      float val = tf * IDF;
      
      res.mIDs[id] = wordId;
      res.mVals[id]=  val;
      
      norm += val * val;
      id++;
    }
    // Normalizing
    norm = (float)(1.0/Math.sqrt(norm));
    
    for (int i = 0; i < res.mIDs.length; ++i) {
      res.mVals[i] *= norm;
    }
    
    return res;
  }  

  @Override
  public String getName() {
    return "Cosine text";
  }
}
