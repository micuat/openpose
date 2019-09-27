#ifndef OPENPOSE_FILESTREAM_W_UDP_SENDER_HPP
#define OPENPOSE_FILESTREAM_W_UDP_SENDER_HPP

#include <openpose/core/common.hpp>
#include <openpose/filestream/udpSender.hpp>
#include <openpose/thread/workerConsumer.hpp>

namespace op
{
    template<typename TDatums>
    class WUdpSender : public WorkerConsumer<TDatums>
    {
    public:
        explicit WUdpSender(const std::shared_ptr<UdpSender>& udpSender);

        virtual ~WUdpSender();

        void initializationOnThread();

        void workConsumer(const TDatums& tDatums);

    private:
        const std::shared_ptr<UdpSender> spUdpSender;

        DELETE_COPY(WUdpSender);
    };
}





// Implementation
#include <openpose/utilities/pointerContainer.hpp>
namespace op
{
    template<typename TDatums>
    WUdpSender<TDatums>::WUdpSender(const std::shared_ptr<UdpSender>& udpSender) :
        spUdpSender{udpSender}
    {
    }

    template<typename TDatums>
    WUdpSender<TDatums>::~WUdpSender()
    {
    }

    template<typename TDatums>
    void WUdpSender<TDatums>::initializationOnThread()
    {
    }

    template<typename TDatums>
    void WUdpSender<TDatums>::workConsumer(const TDatums& tDatums)
    {
        try
        {
            if (checkNoNullNorEmpty(tDatums))
            {
                // Debugging log
                dLog("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
                // Profiling speed
                const auto profilerKey = Profiler::timerInit(__LINE__, __FUNCTION__, __FILE__);
				// Save body/face/hand keypoints to JSON file
				const auto& tDatumFirstPtr = (*tDatums)[0];
				const auto baseFileName = (!tDatumFirstPtr->name.empty() ? tDatumFirstPtr->name
					: std::to_string(tDatumFirstPtr->id)) + "_keypoints";
				const bool humanReadable = false;
				for (auto i = 0u; i < tDatums->size(); i++)
				{
					const auto& tDatumPtr = (*tDatums)[i];
					// const auto fileName = baseFileName;
					const auto fileName = baseFileName + (i != 0 ? "_" + std::to_string(i) : "");

					// Pose IDs from long long to float
					Array<float> poseIds{ tDatumPtr->poseIds };

					const std::vector<std::pair<Array<float>, std::string>> keypointVector{
						// Pose IDs
						std::make_pair(poseIds, "person_id"),
						// 2D
						std::make_pair(tDatumPtr->poseKeypoints, "pose_keypoints_2d"),
						std::make_pair(tDatumPtr->faceKeypoints, "face_keypoints_2d"),
						std::make_pair(tDatumPtr->handKeypoints[0], "hand_left_keypoints_2d"),
						std::make_pair(tDatumPtr->handKeypoints[1], "hand_right_keypoints_2d"),
						// 3D
						std::make_pair(tDatumPtr->poseKeypoints3D, "pose_keypoints_3d"),
						std::make_pair(tDatumPtr->faceKeypoints3D, "face_keypoints_3d"),
						std::make_pair(tDatumPtr->handKeypoints3D[0], "hand_left_keypoints_3d"),
						std::make_pair(tDatumPtr->handKeypoints3D[1], "hand_right_keypoints_3d")
					};

					// Save keypoints
					spUdpSender->send2DJoints(keypointVector, tDatumPtr->poseCandidates);
				}
				// Profiling speed
				Profiler::timerEnd(profilerKey);
				Profiler::printAveragedTimeMsOnIterationX(profilerKey, __LINE__, __FUNCTION__, __FILE__);
				// Debugging log
				dLog("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);

				// Send though UDP communication
#ifdef USE_3D_ADAM_MODEL
                const auto& tDatumPtr = (*tDatums)[0];
                if (!tDatumPtr->poseKeypoints3D.empty())
                {
                    const auto& adamPose = tDatumPtr->adamPose; // Eigen::Matrix<double, 62, 3, Eigen::RowMajor>
                    const auto& adamTranslation = tDatumPtr->adamTranslation; // Eigen::Vector3d(3, 1)
                    const auto adamFaceCoeffsExp = tDatumPtr->adamFaceCoeffsExp; // Eigen::VectorXd resized to (200, 1)
                    //const float mouth_open = tDatumPtr->mouthOpening; // tDatumPtr->mouth_open;
                    //const float leye_open = tDatumPtr->rightEyeOpening; // tDatumPtr->leye_open;
                    //const float reye_open = tDatumPtr->leftEyeOpening; // tDatumPtr->reye_open;
                    //const float dist_root_foot = Datum.distanceRootFoot; // tDatumPtr->dist_root_foot;
                    // m_adam_t:
                    //     1. Total translation (centimeters) of the root in camera/global coordinate representation.
                    // m_adam_pose:
                    //     1. First row is global rotation, in AngleAxis representation. Radians (not degrees!)
                    //     2. Rest are joint-angles in Euler-Angle representation. Degrees.
                    spUdpSender->sendJointAngles(
                        adamPose.data(), adamPose.rows(), adamTranslation.data(), adamFaceCoeffsExp.data(),
                        adamFaceCoeffsExp.rows());
                }
#endif
                // Profiling speed
                Profiler::timerEnd(profilerKey);
                Profiler::printAveragedTimeMsOnIterationX(profilerKey, __LINE__, __FUNCTION__, __FILE__);
                // Debugging log
                dLog("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
            }
        }
        catch (const std::exception& e)
        {
            this->stop();
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    COMPILE_TEMPLATE_DATUM(WUdpSender);
}

#endif // OPENPOSE_FILESTREAM_W_UDP_SENDER_HPP
