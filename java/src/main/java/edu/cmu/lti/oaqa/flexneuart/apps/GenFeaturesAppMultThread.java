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
  package edu.cmu.lti.oaqa.flexneuart.apps;

  import java.io.*;
import java.util.*;

import edu.cmu.lti.oaqa.flexneuart.cand_providers.CandidateEntry;
import edu.cmu.lti.oaqa.flexneuart.utils.DataEntryFields;
import no.uib.cipr.matrix.DenseVector;
  
class GenFeaturesAppImpl extends BaseQueryApp {    
    @Override
    void addOptions() {
      boolean onlyLucene    = false;
      boolean multNumRetr   = true;
      boolean useQRELs      = true;
      boolean useThreadQty  = true;    
      addCandGenOpts(onlyLucene, 
                     multNumRetr,
                     useQRELs,
                     useThreadQty);

      addResourceOpts();
      
      boolean useIntermModel = true, 
              useFinalModel = false; // There is no second-stage re-ranking here!
      addLetorOpts(useIntermModel, useFinalModel);
            
      mOptions.addOption(CommonParams.FEATURE_FILE_PARAM,   null, true, CommonParams.FEATURE_FILE_DESC);    
    }

    @Override
    void procCustomOptions() {
      mOutPrefix = mCmd.getOptionValue(CommonParams.FEATURE_FILE_PARAM);
      if (mOutPrefix == null) 
        showUsageSpecify(CommonParams.FEATURE_FILE_PARAM);
      if (null == mExtrTypeFinal)
        showUsageSpecify(CommonParams.EXTRACTOR_TYPE_FINAL_PARAM);
      if (null == mQrels)
        showUsageSpecify(CommonParams.QREL_FILE_PARAM);
    }

    @Override
    void init() throws IOException {
      for (int numRet : mNumRetArr) {
        String outFile = outFileName(mOutPrefix, numRet);
        mhOutFiles.put(numRet, new BufferedWriter(new FileWriter(new File(outFile))));
      }    
    }

    @Override
    void fin() throws IOException {
      for (Map.Entry<Integer, BufferedWriter> e: mhOutFiles.entrySet()) {
        e.getValue().close();
      }
    }

    @Override
    void procResults(String runId,
                     String queryId,
                     DataEntryFields queryFields,
                     CandidateEntry[] scoredDocs, int numRet, Map<String, DenseVector> docFeats)
        throws IOException {
      BufferedWriter featOut = mhOutFiles.get(numRet);
      if (null == featOut) 
        throw new RuntimeException("Bug, output file is not init. for numRet=" + numRet);

      for (CandidateEntry e : scoredDocs) {
        String label = e.mRelevGrade + "";
        String docId = e.mDocId;
        DenseVector vect = docFeats.get(docId);
        
        StringBuffer sb = new StringBuffer();
        
        sb.append(label + " ");
        sb.append("qid:" + queryId);
        
        for (int fn = 0; fn < vect.size(); ++fn)
          // Note that feature numbers should start from 1 or else some libraries like RankLib will not work correctly!
          sb.append(" " + (fn+1) + ":" + vect.get(fn));
        
        featOut.write(sb.toString());
        featOut.newLine();
      }
    }
    
    String outFileName(String filePrefix, int numRet) {
      return filePrefix + "_" + numRet + ".feat";
    }

    private final HashMap<Integer, BufferedWriter>  mhOutFiles = new HashMap<Integer, BufferedWriter>();
    private String mOutPrefix;
  }

  public class GenFeaturesAppMultThread {

    public static void main(String[] args) {
      try {
        (new GenFeaturesAppImpl()).run("Feature-geneartion application", args);
      } catch(Exception e) {
        e.printStackTrace();
        System.err.println("Terminating due to an exception: " + e);
        System.exit(1);
      } 
    }  
  }
