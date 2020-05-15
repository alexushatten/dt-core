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
    ros::NodeHandle nh_;
    image_transport::ImageTransport it_;
    image_transport::Subscriber image_sub_;
    image_transport::Publisher image_pub_;

    public:
        DecoderNodelet() 
          : it_(nh_)
        {}

    private:
        virtual void onInit()
        {
        image_transport::TransportHints hints("compressed",ros::TransportHints().tcpNoDelay(true), getPrivateNodeHandle());
        image_pub_ = it_.advertise("image/raw", 1);
        image_sub_ = it_.subscribe("image", 1, &DecoderNodelet::imageCallback, this, hints);
        
        }
        void imageCallback(const sensor_msgs::ImageConstPtr& msg){image_pub_.publish(msg);}
    };

    PLUGINLIB_EXPORT_CLASS(image_processing::DecoderNodelet, nodelet::Nodelet)
}
