
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .serializers import ImageUploadSerializer
from .inference import predict_action

class ActionRecognition(APIView):
    def post(self, request, format=None):
        serializer = ImageUploadSerializer(data=request.data)
        
        if serializer.is_valid():
            image = serializer.validated_data['image']
            predicted_action = predict_action(image)
            return Response({'predicted_action': predicted_action}, status=status.HTTP_200_OK)
        
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
