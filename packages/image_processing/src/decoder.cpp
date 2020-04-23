#include <ros/ros.h>
#include "image_transport/image_transport.h"
#include <pluginlib/class_loader.h>
#include <nodelet/nodelet.h>
#include <pluginlib/class_list_macros.h>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>


namespace image_processing
{

    class DecoderNodelet : public nodelet::Nodelet
    {
    public:
        DecoderNodelet()
        {}

    private:
        virtual void onInit()
        {
        ros::NodeHandle& private_nh = getPrivateNodeHandle();
        //image_transport::ImageTransport it(nh);
        pub = private_nh.advertise<sensor_msgs::Image>("image/raw", 1);
        sub = private_nh.subscribe("compressed_image", 10, &DecoderNodelet::imageCallback, this);
        }


        void imageCallback(const sensor_msgs::CompressedImageConstPtr& msg)
        {
        cv::Mat image = cv::imdecode(cv::Mat(msg->data),1);//convert compressed image data to cv::Mat
        cv_bridge::CvImage raw_msg; //= cv_bridge::CvImage(msg->header, "bgr8", image.toImageMsg();
        raw_msg.header = msg->header;
        raw_msg.header.seq = msg->header.seq;
        raw_msg.encoding = sensor_msgs::image_encodings::BGR8 ; // Or whatever
        raw_msg.image = image;
        pub.publish(raw_msg.toImageMsg());
        }
        ros::Publisher pub;
        ros::Subscriber sub;

    };

    PLUGINLIB_EXPORT_CLASS(image_processing::DecoderNodelet, nodelet::Nodelet)
}
