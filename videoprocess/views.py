from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.http import JsonResponse
import torch
import torchvision
from .utils import Modeldevelopment 
import tempfile
class VideoApi(APIView):
    # Model parameters
    frames = 32
    period = 1  # 2
    batch_size = 20

    # Load the Efficient Model
    ef_model = torchvision.models.video.r2plus1d_18(pretrained=False)
    ef_model.fc = torch.nn.Linear(ef_model.fc.in_features, 1)
    device = torch.device("cpu")
    ef_checkpoint = torch.load('/Users/jayanth/Desktop/backend/backend/model/r2plus1d_18_32_2_pretrained.pt', map_location="cpu")
    ef_state_dict_cpu = {k[7:]: v for (k, v) in ef_checkpoint['state_dict'].items()}
    ef_model.load_state_dict(ef_state_dict_cpu)

    # Load the Segmentation Model
    seg_model = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=True)
    seg_model.classifier[-1] = torch.nn.Conv2d(seg_model.classifier[-1].in_channels, 1, kernel_size=seg_model.classifier[-1].kernel_size)
    seg_checkpoint = torch.load('/Users/jayanth/Desktop/backend/backend/model/DeeplabV3 Resnet101 Best.pt', map_location="cpu")
    seg_state_dict_cpu = {k[7:]: v for (k, v) in seg_checkpoint['state_dict'].items()}
    seg_model.load_state_dict(seg_state_dict_cpu)

    # Instance of Modeldevelopment
    mod = Modeldevelopment()

    def post(self, request, *args, **kwargs):
        """
        Handles POST requests for video processing.
        """
        try:
            # Extract the video file from the request
            video_file = request.FILES.get('video')
            if not video_file:
                return Response({"error": "No video file provided"}, status=status.HTTP_400_BAD_REQUEST)

            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                for chunk in video_file.chunks():
                    tmp_file.write(chunk)
                tmp_file_path = tmp_file.name

            result = self.mod.final_results(seg_model=self.seg_model, ef_model=self.ef_model, input_folder=tmp_file_path, output_folder='backend/backend/model/output')
            result_dict = result if isinstance(result, dict) else result.__dict__
            return JsonResponse({"message": "Processing complete", "results": result_dict}, status=status.HTTP_200_OK)

        

        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
