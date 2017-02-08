/opt/hadoop-1.2.1/bin/hadoop fs -rmr /giraph/ou*

#-ca giraph.zkList=bdp-10:21281 \
#-ca giraph.useInputSplitLocality=false \
#-ca giraph.isStaticGraph=true \

/opt/hadoop-1.2.1/bin/hadoop \
jar /opt/giraph/giraph-examples/target/giraph-examples-1.3.0-SNAPSHOT-for-hadoop-1.2.1-jar-with-dependencies.jar \
org.apache.giraph.GiraphRunner org.apache.giraph.examples.SimpleShortestPathsComputation \
-ca giraph.waitTaskDoneTimeoutMs=900000000 \
-ca giraph.zKMinSessionTimeout=800000000 \
-ca giraph.zKMaxSessionTimeout=900000000 \
-ca giraph.nettyCompressionAlgorithm=SNAPPY \
-ca giraph.useOutOfCoreGraph=true \
-ca giraph.enableFlowControlInput=true \
-ca giraph.waitForPerWorkerRequests=true \
-ca giraph.useBigDataIOForMessages=true \
-ca giraph.maxPartitionsInMemory=2 \
-ca giraph.maxNumberOfSupersteps=50 \
-ca giraph.maxMasterSuperstepWaitMsecs=60000000 \
-ca giraph.nettyServerThreads=4 \
-ca giraph.nettyServerExecutionThreads=2 \
-ca giraph.nettyClientExecutionThreads=4 \
-ca giraph.nettyClientThreads=2 \
-vif org.apache.giraph.io.formats.JsonLongDoubleFloatDoubleVertexInputFormat \
-vip  /giraph/uk-2014.giraph \
-vof org.apache.giraph.io.formats.IdWithValueTextOutputFormat \
-op /giraph/out15 \
-w 100
