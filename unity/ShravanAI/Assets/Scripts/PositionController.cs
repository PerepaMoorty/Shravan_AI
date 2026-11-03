using UnityEngine;

public class PositionController : MonoBehaviour
{
    Vector3 currPosition = Vector3.zero;
    void Awake() => currPosition = transform.position;
    void FixedUpdate() => transform.localPosition = currPosition;    
}