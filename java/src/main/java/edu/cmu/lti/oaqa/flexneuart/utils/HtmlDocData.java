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
package edu.cmu.lti.oaqa.flexneuart.utils;

public class HtmlDocData {
  public final String mTitle;
  public final String mBodyText;
  public final String mLinkText;
  public final String mAllText;
  
  public HtmlDocData(String title, String bodyText, String linkText, String allText) {
    this.mTitle = title;
    this.mBodyText = bodyText;
    this.mLinkText = linkText;
    this.mAllText = allText;
  }
  
}
