import math

class Retrieve:

    #==============================================================================
    # Constructor and public methods
    
    def __init__(self, index, termWeighting):
        ''' Retrieve constructor.

        Create new Retrieve object storing index and termWeighting scheme.

        Args:
            index ({int: {str: int}}): A two-level dictionary structure, map-pingtermstodoc-idstocounts, representing the document collection.
            termWeighting (str): Weighting scheme "LABEL" (LABEL in {binary, tf, tfidf}, default: binary)

        '''
        self.index = index
        self.termWeighting = termWeighting

        self.resetCollectionSize()
        
    def resetCollectionSize(self):
        ''' Reset collection size.

        Once the index object has been modified, this method should be called to recalculate the collection size.

        '''
        # A globally used index statistics for tf.idf. (Lazy loading)
        self.__collectionSize = -1
        # Record the id of the index object corresponding to collectionSize so that collectionSize can be auto-update when the index is changed externally.
        self.__collectionSizeTarget = -1

    def forQuery(self, query):
        ''' Method performing retrieval for specified query.

        Args:
            query ({str: int}): The query that needs to be retrieved.

        Returns:
            A list of retrieved docid.
        '''
        # For debug only
        # print("Q:", query)

        if self.termWeighting == 'binary':
            return self.__forQueryBinary(query)
        elif self.termWeighting == 'tf':
            return self.__forQueryTf(query)
        elif self.termWeighting == 'tfidf':
            return self.__forQueryTfidf(query)
        
        print("*** Warning: Invalid weighting method (\"%s\"), binary weighting is used by default ***" % self.termWeighting)
        return self.__forQueryBinary(query)

    #==============================================================================
    # Retrieval methods under different weighting method

    def __forQueryBinary(self, query):
        ''' Binary weighting retrieval 

        Args:
            query ({str: int}): The query that needs to be retrieval.

        Returns:
            A list of ranked query-related docid.
        '''
        # return self.__rankByBinary(query)
        return self.__rankByCosSim(query, self.__binaryWeighting)

    def __forQueryTf(self, query):
        ''' TF weighting retrieval 

        Args:
            query ({str: int}): The query that needs to be retrieval.

        Returns:
            A list of ranked query-related docid.
        '''
        return self.__rankByCosSim(query, self.__tfWeighting)

    def __forQueryTfidf(self, query):
        ''' TF.IDF weighting retrieval 

        Args:
            query ({str: int}): The query that needs to be retrieval.

        Returns:
            A list of ranked query-related docid.
        '''
        return self.__rankByCosSim(query, self.__tfidfWeighting)

    #==============================================================================
    # Auxiliary methods

    def __findRelated(self, query):
        ''' Find query-related documents.

        Use Boolean search find query-related documents, where term in the query need to be present in the document.

        Args:
            query ({str: int}): The query that needs to be retrieval.

        Returns:
            A list of query-related docid.
        '''
        result = set()
        for queryTerm in query.keys():
            if queryTerm in self.index:
                relatesIndex = self.index[queryTerm]
                relatesDocids = relatesIndex.keys()
                result.update(relatesDocids)
        return result

    def __rankByCosSim(self, query, weightingCallback):
        ''' Rank query-related documents on a cosine distance based model.

        Rank query-related documents by computing the cosine similiarity, where the weighting method of terms in the document can be specified by callback

        Args:
            query ({str: int}): The query that needs to be retrieval.
            weightingCallback ((term, freq) -> float): Callback method for calculating the weight of terms in the document.

        Returns:
            A list of ranked query-related docid.
        '''
        # Find related docs
        relatesDocids = self.__findRelated(query)

        # Calc sum square of d (ssd)
        ssdDict = dict()
        for docid in relatesDocids:
            ssdDict[docid] = 0
        for term in self.index.keys():
            termIndex = self.index[term]
            for docid in termIndex:
                if docid not in relatesDocids:
                    continue
                d = weightingCallback(term, termIndex[docid])
                ssdDict[docid] += d * d

        # Calc sum of q * d (sqd) & document similarity
        docsSimilarity = []
        for docid in relatesDocids:
            sqd = 0
            for term in query.keys():
                if term in self.index:
                    termIndex = self.index[term]
                    if docid not in termIndex:
                        continue
                    d = weightingCallback(term, termIndex[docid])
                    q = weightingCallback(term, query[term])
                    sqd += q * d
            sim = sqd / math.sqrt(ssdDict[docid])
            docsSimilarity.append((docid, sim))
        
        # Sort result
        docsSimilarity.sort(key=lambda i : i[1], reverse=True)
        sortedResult = [ i[0] for i in docsSimilarity ]

        return sortedResult

    #==============================================================================
    # Weighting callbacks

    def __binaryWeighting(self, term, freq):
        ''' Binary weighting callback.

        Weighting a term in a specific document or query by Binary weighting.

        Args:
            term (str): Weighting term.
            freq (int): Term frequency in the specific context.

        Returns:
            Term weight in the document.
        '''
        return 1

    def __tfWeighting(self, term, freq):
        ''' TF weighting callback.

        Weighting a term in a specific document or query by the Frequency of term.

        Args:
            term (str): Weighting term.
            freq (int): Term frequency in the specific context.

        Returns:
            Term weight in the document.
        '''
        tf = freq
        return tf

    def __tfidfWeighting(self, term, freq):
        ''' TF.IDF weighting callback.

        Weighting a term in a specific document or query by the Frequency in document vs in collection.

        Args:
            term (str): Weighting term.
            freq (int): Term frequency in the specific context.

        Returns:
            Term weight in the document.
        '''
        # Init collection size (Lazy loading)
        if(self.__collectionSizeTarget != id(self.index)):
            print('initializing collection size... (Lazy loading)')
            docSet = set()
            for term in self.index.keys():
                docSet.update(self.index[term].keys())
            self.__collectionSize = len(docSet)
            self.__collectionSizeTarget = id(self.index)
            print('Collection size:', self.__collectionSize)

        # Calc weight
        tf = self.__tfWeighting(term, freq)
        idf = math.log(self.__collectionSize / len(self.index[term]))

        tfidf = tf * idf
        return tfidf
