# RAG Strategy

## Strategy of the RAG System

## Data Ingestion

**Objective**: Efficiently extract text and detailed descriptions from documents filled with charts, diagrams, and images using Vision LLM, optimizing for time, cost, and accuracy.

As the input document has a complex, Power-Point-like layout, I decided to use Vision LLMs to extract data instead of OCR. Currently, OCRs are unable to fetch data in the correct layout, as they fetch text line by line, which misses the exact layout or sequence of text.

1. **Conversion to Images**: 
   - Convert each page of the document into an image format. This is essential for processing visual elements such as charts and diagrams effectively.
   
2. **Batch Processing**: 
   - Process these images in batches to reduce time and cost. By batching the images, the system can handle multiple pages simultaneously, reducing the overall processing time.
   - Utilize asynchronous function calls to run the LLM concurrently, extracting data from each batch simultaneously. This further reduces latency. Asynchronous processing allows multiple requests to be handled at the same time, speeding up the extraction process significantly.

3. **Detailed Descriptions**: 
   - Request the LLM to generate detailed descriptions for each image. These descriptions provide additional context and details about the visual content.
   - These descriptions help refine extracted data, especially for complex elements like tables which might not be correctly formatted initially. If the initial extraction misses certain details or formats, the detailed descriptions can provide the necessary context to correct and enhance the extracted data.

## Splitting Document into Smaller Chunks

**Objective**: Maintain context within chunks while ensuring they are of a manageable size for embedding generation and further processing.

1. **Recursive Character Text Splitter**: 
   - Utilize Langchain’s RecursiveCharacterTextSplitter with a token limit and priority separators ["\n\n", ".", "\n"]. 
   - This ensures chunks are split at logical points, maintaining context and coherence within each chunk.
   - Through experimentation, a chunk size of 100 tokens was selected. This size was found to balance the need for context with the need to manage chunk size for embedding purposes. It helps in maintaining some context in each chunk, making it easier for the model to understand and process the information.

## Embedding Generation and Upserting Vectors into Vector DB

**Objective**: Generate and store contextual embeddings for each chunk, ensuring efficient and accurate retrieval during inference.

1. **Embedding Generation**: 
   - Generate embeddings for each chunk using the "voyage-large-2-instruct" model from voyageai, known for its performance in general-purpose tasks. This model provides high-quality embeddings that capture the context and meaning of the text.
   
2. **Metadata Storage**: 
   - Store chunk_id, page_no, chunk_text, and page_description as metadata for each chunk. This metadata is essential for organizing and retrieving the chunks efficiently during the inference stage.
   
3. **Vector Database**: 
   - Used Pinecone for vector storage due to its low latency and hybrid search functionality. Pinecone’s capabilities ensure that the embeddings can be retrieved quickly and accurately, supporting real-time query processing.
   
4. **Future Improvements**: 
   - Generate sparse embeddings of the chunks and contextual embeddings of image descriptions. This dual embedding approach can enhance the retrieval accuracy by providing multiple perspectives on the data.
   - Experiment with different combinations to determine the best retrieval accuracy. This iterative process will help in finding the optimal embedding strategy for various types of data.

## Inference

**Objective**: Efficiently retrieve relevant information in response to a query and generate accurate answers using the contextual data.

1. **Query Embedding**: 
   - Generate embeddings for the input query using the same embedding model. This step ensures that the query is represented in the same vector space as the chunks, facilitating accurate similarity calculations.
   
2. **Chunk Retrieval**: 
   - Retrieve the top 5 relevant chunks using cosine similarity for similarity calculations, optimized for voyageai models.
   - Filter out relevant chunks by comparing the similarity score with a pre-selected threshold (68% similarity). This threshold, determined through experimentation, ensures that only highly relevant chunks are considered for the final answer.
   
3. **Metadata Extraction**: 
   - For the top 3 vectors that pass the similarity score filter, extract chunk text, page number, page description, and the image of the most relevant chunks. This comprehensive extraction ensures that all necessary context is available for answer generation.
   
4. **Structured Input for Model**: 
   - Pass all this information in a well-structured format to the prompt.
   - Ensure the model receives comprehensive details related to the query, allowing it to derive context from both the chunk and the associated image and description.
   
5. **Answer Generation**: 
   - Use GPT-4O for generating answers from the relevant context. This model has been selected for its superior performance in generating detailed and accurate responses based on the provided context.

## Evaluation Strategy

**Objective**: To ensure the generated answers are accurate, relevant, and free from hallucinations, a two-level evaluation strategy is employed. This strategy utilizes both automated similarity checks and manual review through a judge LLM.

### Two-Level Evaluation

**1. Cosine Similarity Check**
- **Embedding Generation**: 
  - Generate embeddings for both the generated answer and the retrieved relevant context. These embeddings capture the semantic meaning of the text, allowing for an accurate comparison.
  
- **Similarity Measurement**: 
  - Measure the cosine similarity between the embeddings of the generated answer and the relevant context. Cosine similarity provides a metric for how closely the two vectors align, indicating the relevance of the generated answer to the context.
  
- **Threshold Check**: 
  - Compare the similarity score against a pre-defined threshold. This threshold is determined through experimentation to balance precision and recall. If the similarity score exceeds the threshold, the answer is considered relevant and is passed to the end user.
  
- **Answer Re-generation**: 
  - If the similarity score does not meet the threshold, tweak the inference prompt slightly and add more context. This iterative approach helps in refining the answer by providing additional information to the model.
  
**2. Judge LLM Verification**
- **Manual Review by Judge LLM**: 
  - If the first evaluation check fails, pass the generated answer to a judge LLM. This LLM is a smaller, efficient model specifically designed for evaluating the relevance and accuracy of the generated answers.
  
- **Context and Image Verification**: 
  - Feed the judge LLM with the generated answer and the relevant context, potentially including images. This comprehensive input ensures the judge LLM has all necessary information to evaluate the answer accurately.
  
- **Hallucination and Relevance Check**: 
  - The judge LLM checks the answer for hallucinations and verifies its relevance to the provided context. It ensures that the answer is not only accurate but also directly related to the user's query and the given context.
  
- **Final Decision**: 
  - Based on the judge LLM’s evaluation, decide whether to pass the answer to the end user or to re-generate the answer with additional context. This step provides an additional layer of assurance, ensuring high-quality responses.
