#ifdef USE_ASIO
    #include <iostream>
    #include <asio.hpp>
#endif
#ifdef USE_EIGEN
    #include <Eigen/Core>
#endif
#include <openpose/filestream/fileStream.hpp>
#include <openpose/filestream/udpSender.hpp>

#include <openpose/filestream/jsonOfstream.hpp>
#include <openpose/utilities/fastMath.hpp>

namespace op
{
    #ifdef USE_ASIO
        class UdpClient
        {
        public:
            UdpClient(const std::string& host, const std::string& port) :
                mIoService{},
                mUdpSocket{mIoService, asio::ip::udp::endpoint(asio::ip::udp::v4(), 0)}
            {
                try
                {
                    asio::ip::udp::resolver resolver{mIoService};
                    asio::ip::udp::resolver::query query{asio::ip::udp::v4(), host, port};
                    asio::ip::udp::resolver::iterator iter = resolver.resolve(query);
                    mUdpEndpoint = *iter;
                }
                catch (const std::exception& e)
                {
                    error(e.what(), __LINE__, __FUNCTION__, __FILE__);
                }
            }

            ~UdpClient()
            {
                try
                {
                    mUdpSocket.close();
                }
                catch (const std::exception& e)
                {
                    errorDestructor(e.what(), __LINE__, __FUNCTION__, __FILE__);
                }
            }

            void send(const std::string& msg)
            {
                try
                {
                    mUdpSocket.send_to(asio::buffer(msg, msg.size()), mUdpEndpoint);
                    // std::cout << "sent data: " << msg << std::endl;
                }
                catch (const std::exception& e)
                {
                    error(e.what(), __LINE__, __FUNCTION__, __FILE__);
                }
            }

        private:
            asio::io_service mIoService;
            asio::ip::udp::socket mUdpSocket;
            asio::ip::udp::endpoint mUdpEndpoint;
        };


        std::string vectorToJson(const float x, const float y, const float z)
        {
            return std::string{"{"} +
                "\"x\":" + std::to_string(x) + "," +
                "\"y\":" + std::to_string(y) + "," +
                "\"z\":" + std::to_string(z) + "}";
        }
    #endif

    struct UdpSender::ImplUdpSender
    {
        #ifdef USE_ASIO
            // Used when increasing spCaffeNets
            UdpClient mUdpClient;

            ImplUdpSender(const std::string& udpHost, const std::string& udpPort) :
                mUdpClient(udpHost, udpPort)
            {
            }
        #endif
    };

    UdpSender::UdpSender(const std::string& udpHost, const std::string& udpPort)
        #ifdef USE_ASIO
            : spImpl{new ImplUdpSender{udpHost, udpPort}}
        #endif
    {
        try
        {
            // error("UdpSender (`--udp_host` and `--udp_port` flags) buggy and not working yet, but we are"
            //       "working on it! Coming soon!", __LINE__, __FUNCTION__, __FILE__);
            #if !defined(USE_ASIO) || !defined(USE_EIGEN)
                error("Both `WITH_ASIO` and `WITH_EIGEN` flags must be enabled in CMake for UDP sender.",
                      __LINE__, __FUNCTION__, __FILE__);
                UNUSED(udpHost);
                UNUSED(udpPort);
            #endif
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    UdpSender::~UdpSender()
    {
    }

	void UdpSender::send2DJoints(const std::vector<std::pair<Array<float>, std::string>>& keypointVector,
		const std::vector<std::vector<std::array<float, 3>>> & candidates) {

		JsonOfstream jsonOfstream("test.json", true);
		// Sanity check
		for (const auto& keypointPair : keypointVector)
			if (!keypointPair.first.empty() && keypointPair.first.getNumberDimensions() != 3
				&& keypointPair.first.getNumberDimensions() != 1)
				error("keypointVector.getNumberDimensions() != 1 && != 3.", __LINE__, __FUNCTION__, __FILE__);
		// Add people keypoints
		jsonOfstream.key("people");
		jsonOfstream.arrayOpen();
		// Ger max numberPeople
		auto numberPeople = 0;
		for (auto vectorIndex = 0u; vectorIndex < keypointVector.size(); vectorIndex++)
			numberPeople = fastMax(numberPeople, keypointVector[vectorIndex].first.getSize(0));
		for (auto person = 0; person < numberPeople; person++)
		{
			jsonOfstream.objectOpen();
			for (auto vectorIndex = 0u; vectorIndex < keypointVector.size(); vectorIndex++)
			{
				const auto& keypoints = keypointVector[vectorIndex].first;
				const auto& keypointName = keypointVector[vectorIndex].second;
				const auto numberElementsPerRaw = keypoints.getSize(1) * keypoints.getSize(2);
				jsonOfstream.key(keypointName);
				jsonOfstream.arrayOpen();
				// Body parts
				if (numberElementsPerRaw > 0)
				{
					const auto finalIndex = person * numberElementsPerRaw;
					for (auto element = 0; element < numberElementsPerRaw - 1; element++)
					{
						jsonOfstream.plainText(keypoints[finalIndex + element]);
						jsonOfstream.comma();
					}
					// Last element (no comma)
					jsonOfstream.plainText(keypoints[finalIndex + numberElementsPerRaw - 1]);
				}
				// Close array
				jsonOfstream.arrayClose();
				if (vectorIndex < keypointVector.size() - 1)
					jsonOfstream.comma();
			}
			jsonOfstream.objectClose();
			if (person < numberPeople - 1)
			{
				jsonOfstream.comma();
				jsonOfstream.enter();
			}
		}
		// Close bodies array
		jsonOfstream.arrayClose();

		std::string data = jsonOfstream.getStream()->str();
		spImpl->mUdpClient.send(data);
	}

	void UdpSender::sendJointAngles(const double* const adamPosePtr, const int adamPoseRows,
                                    const double* const adamTranslationPtr,
                                    const double* const adamFaceCoeffsExpPtr, const int faceCoeffRows)
    {
        #if defined(USE_ASIO) && defined(USE_EIGEN)
            try
            {
                if (adamPosePtr != nullptr && adamTranslationPtr != nullptr && adamFaceCoeffsExpPtr != nullptr)
                {
                    const Eigen::Map<const Eigen::Vector3d> adamTranslation(adamTranslationPtr);
                    const Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> adamPose(
                        adamPosePtr, adamPoseRows, 3);
                    const Eigen::Map<const Eigen::VectorXd> adamFaceCoeffsExp(adamFaceCoeffsExpPtr, faceCoeffRows);

                    const std::string prefix = "AnimData:";
                    const std::string totalPositionString = "\"totalPosition\":"
                        + vectorToJson(adamTranslation(0), adamTranslation(1), adamTranslation(2));
                    std::string jointAnglesString = "\"jointAngles\":[";
                    for (int i = 0; i < adamPoseRows; i++)
                    {
                        jointAnglesString += vectorToJson(adamPose(i, 0), adamPose(i, 1), adamPose(i, 2));
                        if (i != adamPoseRows - 1)
                        {
                            jointAnglesString += ",";
                        }
                    }
                    jointAnglesString += "]";

                    std::string facialParamsString = "\"facialParams\":[";
                    for (int i = 0; i < faceCoeffRows; i++)
                    {
                        facialParamsString += std::to_string(adamFaceCoeffsExp(i));
                        if (i != faceCoeffRows - 1)
                        {
                            facialParamsString += ",";
                        }
                    }
                    facialParamsString += "]";

                    // facialParamsString + std::to_string(mouth_open) + "," + std::to_string(leye_open) + "," + std::to_string(reye_open) + "]";

                    // std::string rootHeightString = "\"rootHeight\":" + std::to_string(dist_root_foot);

                    const std::string data = prefix + "{" + facialParamsString
                                           + "," + totalPositionString
                                           + "," + jointAnglesString + "}";

                    spImpl->mUdpClient.send(data);
                }
            }
            catch (const std::exception& e)
            {
                error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            }
        #else
            UNUSED(adamPosePtr);
            UNUSED(adamPoseRows);
            UNUSED(adamTranslationPtr);
            UNUSED(adamFaceCoeffsExpPtr);
            UNUSED(faceCoeffRows);
        #endif
    }
}
