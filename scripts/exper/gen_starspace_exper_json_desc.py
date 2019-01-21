#!/usr/bin/env python
import os, sys, json

# Input location:
# inpRootDir / colName / starspace / <file names>

# Output location for descriptor files:
# Individual extractor files
# outRootDir / colName / starspace / <specific suffix>.json
# outRootDir / colName / starspace.desc

# Relative output location for experimental files:
# starspace / <specific suffix> 

embedRootDir = sys.argv[1]
colName      = sys.argv[2]
outRootDir   = sys.argv[3]

outDescDir = os.path.join(outRootDir, colName)
outJsonDir = os.path.join(outRootDir, colName, 'starspace')

if not os.path.exists(outDescDir):
  print('Directory does not exist: %s' % outDescDir)
  sys.exit(1)

if not os.path.exists(outJsonDir):
  print('Directory does not exist: %s' % outJsonDir)
  sys.exit(1)

embedDir = os.path.join(embedRootDir, colName, 'starspace') 

with open(os.path.join(outDescDir, 'starspace.desc'), 'w') as of:
  for fn in os.listdir(embedDir):
    if fn.endswith('.query'):  
      fid0 = fn[0:-len('.query')]

      for distType in ['l2', 'cosine']:
        fid = distType + '_' + fid0    
        print(fid)
        jsonDesc = {
                  "extractors" : [
                  {
                    "type" : "TFIDFSimilarity",
                    "params" : {
                      "fieldName" : "text",
                      "similType" : "bm25",
                      "k1"        : "1.2",
                      "b"         : "0.75"
                    }
                  },
                  {
                    "type" : "avgWordEmbed",
                    "params" : {
                      "fieldName" : "text_unlemm",
                      "queryEmbedFile" : "starspace/%s.query" % fid0,
                      "docEmbedFile"   : "starspace/%s.answer" % fid0,
                      "useIDFWeight"   : "True",
                      "useL2Norm"      : "True",
                      "distType"       : distType 
                    }
                  }
                  ]
                  }
        jsonFileName = fid + '.json'
        jsonPath = os.path.join(outJsonDir, jsonFileName)
        of.write('%s dev1 %s\n' % (jsonPath, os.path.join('starspace', fid)))
        
        with open(jsonPath, 'w') as f:
          json.dump(jsonDesc, f)

