from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

from plant_health.services import diagnose_uploaded_image


@csrf_exempt
def diagnose_plant(request):
    if request.method != "POST":
        return JsonResponse({"error": "Only POST method allowed"}, status=405)

    uploaded_image = request.FILES.get("image")
    if not uploaded_image:
        return JsonResponse(
            {"error": "Upload an image file using the 'image' field."},
            status=400,
        )

    diagnosis = diagnose_uploaded_image(uploaded_image)
    status_code = 200 if diagnosis.get("status") in {"ok", "uncertain", "reupload", "model_not_ready"} else 400
    return JsonResponse(diagnosis, status=status_code)
