package org.myorg ;

import java.io.IOException ;
import java.util.* ;
import java.io.* ;

import org.apache.hadoop.conf.Configuration ;
import org.apache.hadoop.filecache.DistributedCache ;
import org.apache.hadoop.fs.FileSystem ;
import org.apache.hadoop.fs.Path ;
import org.apache.hadoop.io.* ;
import org.apache.hadoop.mapred.* ;
import org.apache.hadoop.mapred.Reducer ;

public class KMeans {
    public static String input_directory ;
    public static String output_directory ;
    public static List<Double> clusterCenters = new ArrayList<Double>() ;
    
    public static void main(String[] args) throws Exception {
	k_means(args) ;
    }
    
    public static void k_means(String[] args) throws Exception {
	input_directory = args[0] ;
	output_directory = args[1] ;
	String input = input_directory ;
	String output = output_directory + System.nanoTime() ;
	String previous_output = output ;
	
	boolean isClustered = false, firstPass = true ;
	while(!isClustered) {
	    JobConf jobConf = new JobConf(KMeans.class) ;
	    if(firstPass) {
		// first pass, pass the initial centroids
		Path centroid_path = new Path(input + "/centroid.csv") ;
		// add to the disributed cache using the absolute path name (toUri())
		DistributedCache.addCacheFile(centroid_path.toUri(), jobConf) ;
		
	    } else {
		// pass the previously calculated cluster centers
		Path previous_centroid_path = new Path(previous_output + "/part-00000") ;
		DistributedCache.addCacheFile(previous_centroid_path.toUri(), jobConf) ;
	    }
	    jobConf.setJobName("KMeans") ;
	    
	    jobConf.setMapOutputKeyClass(DoubleWritable.class) ;
	    jobConf.setMapOutputValueClass(DoubleWritable.class) ;
	    
	    jobConf.setOutputKeyClass(DoubleWritable.class) ;
	    jobConf.setOutputValueClass(Text.class) ;
	    
	    jobConf.setMapperClass(Map.class) ;
	    jobConf.setReducerClass(Reduce.class) ;
	    
	    jobConf.setInputFormat(TextInputFormat.class) ;
	    jobConf.setOutputFormat(TextOutputFormat.class) ;
	    
	    FileInputFormat.setInputPaths(jobConf, new Path(input + "/data.csv")) ;
	    FileOutputFormat.setOutputPath(jobConf, new Path(output)) ;
	    
	    try {
		JobClient.runJob(jobConf) ;
	    } catch(IOException ioe) {
		System.err.println(ioe);
	    }
	    Path output_path = new Path(output + "/part-00000") ;
	    FileSystem next_fileSystem = FileSystem.get(new Configuration()) ;
	    BufferedReader next_bufferedReader = new BufferedReader(new InputStreamReader(next_fileSystem.open(output_path))) ;
	    List<Double> next_centers = new ArrayList<Double>() ;
	    String line = next_bufferedReader.readLine() ;
	    while(line != null) {
		String[] split_line = line.split(" |\t") ;
		double center = Double.parseDouble(split_line[0]) ;
		next_centers.add(center) ;
		line = next_bufferedReader.readLine() ;
	    }
	    next_bufferedReader.close() ;
	    
	    String previous_centers_path ;
	    if(firstPass)
		previous_centers_path = input + "/centroid.csv" ;
	    else
		previous_centers_path = previous_output + "/part-00000" ;
	    
	    Path previous_path = new Path(previous_centers_path) ;
	    FileSystem previous_fileSystem = FileSystem.get(new Configuration()) ;
	    BufferedReader previous_bufferedReader = new BufferedReader(new InputStreamReader(previous_fileSystem.open(previous_path))) ;
	    List<Double> previous_centers = new ArrayList<Double>() ;
	    line = previous_bufferedReader.readLine() ;
	    while(line != null) {
		String[] split_line = line.split(" |\t") ;
		double center = Double.parseDouble(split_line[0]) ;
		previous_centers.add(center) ;
		line = previous_bufferedReader.readLine() ;
	    }
	    previous_bufferedReader.close() ;
	    
	    // sort both previous_centers and next_centers and check if they converge
	    Collections.sort(next_centers) ;
	    Collections.sort(previous_centers) ;
	    
	    Iterator<Double> iterator = next_centers.iterator() ;
	    for(double previous_center: previous_centers) {
		double next_center = iterator.next() ;
		if(Math.abs(next_center - previous_center) <= 0.1)
		    isClustered = true ;
		else {
		    isClustered = false ;
		    break ;
		}
	    }
	    
	    firstPass = false ;
	    previous_output = output ;
	    output = output_directory + System.nanoTime() ;
	}
    }
    
    public static class Map extends MapReduceBase implements Mapper<LongWritable, Text, DoubleWritable, DoubleWritable> {
	public void configure(JobConf jobConf) {
	    try {
		Path[] cacheFiles = DistributedCache.getLocalCacheFiles(jobConf) ;
		if(cacheFiles != null && cacheFiles.length > 0) {
		    String line ;
		    clusterCenters.clear() ;
		    BufferedReader bufferedReader = new BufferedReader(new FileReader(cacheFiles[0].toString())) ;
		    try {
			while((line = bufferedReader.readLine()) != null) {
			    String[] split_line = line.split(" |\t") ;
			    clusterCenters.add(Double.parseDouble(split_line[0])) ;
			}
		    } finally {
			bufferedReader.close() ;
		    }
		}
	    } catch(IOException ioe) {
		System.err.println(ioe) ;
	    }
	}
	
	public void map(LongWritable key, Text value, OutputCollector<DoubleWritable, DoubleWritable> output, Reporter reporter) throws IOException {
	    String line = value.toString() ;
	    double point = Double.parseDouble(line) ;
	    double nearestCenter = clusterCenters.get(0), val, min = Double.MAX_VALUE ;
	    for(double center: clusterCenters) {
		val = center - point ;
		if(Math.abs(val) < Math.abs(min)) {
		    nearestCenter = center ;
		    min = val ;
		}
	    }
	    // collect the nearest center and the point
	    output.collect(new DoubleWritable(nearestCenter), new DoubleWritable(point)) ;
	}
    }
    
    public static class Reduce extends MapReduceBase implements Reducer<DoubleWritable, DoubleWritable, DoubleWritable, Text> {
	public void reduce(DoubleWritable key, Iterator<DoubleWritable> values, OutputCollector<DoubleWritable, Text> output, Reporter reporter) throws IOException {
	    double newCenter, sum = 0 ;
	    int numElements = 0 ;
	    String points = "" ;
	    while(values.hasNext()) {
		double center = values.next().get() ;
		points = points + " " + Double.toString(center) ;
		sum += center ;
		numElements++ ;
	    }
	    
	    newCenter = sum/numElements ;
	    // collect the new center and the points
	    output.collect(new DoubleWritable(newCenter), new Text(points)) ;
	}
    }
}
