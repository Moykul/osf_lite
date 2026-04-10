#pragma once

#include "CoreMinimal.h"
#include "Components/ActorComponent.h"
#include "Networking.h"
#include "Sockets.h"
#include "SocketSubsystem.h"
#include "HAL/Runnable.h"
#include "HAL/RunnableThread.h"
#include "ARKitFaceReceiver.generated.h"

/** 61 ARKit blendshape names — index matches packet order */
UENUM(BlueprintType)
enum class EARKitShape : uint8
{
    EyeBlinkLeft        = 0,
    EyeLookDownLeft     = 1,
    EyeLookInLeft       = 2,
    EyeLookOutLeft      = 3,
    EyeLookUpLeft       = 4,
    EyeSquintLeft       = 5,
    EyeWideLeft         = 6,
    EyeBlinkRight       = 7,
    EyeLookDownRight    = 8,
    EyeLookInRight      = 9,
    EyeLookOutRight     = 10,
    EyeLookUpRight      = 11,
    EyeSquintRight      = 12,
    EyeWideRight        = 13,
    JawForward          = 14,
    JawLeft             = 15,
    JawRight            = 16,
    JawOpen             = 17,
    MouthClose          = 18,
    MouthFunnel         = 19,
    MouthPucker         = 20,
    MouthLeft           = 21,
    MouthRight          = 22,
    MouthSmileLeft      = 23,
    MouthSmileRight     = 24,
    MouthFrownLeft      = 25,
    MouthFrownRight     = 26,
    MouthDimpleLeft     = 27,
    MouthDimpleRight    = 28,
    MouthStretchLeft    = 29,
    MouthStretchRight   = 30,
    MouthRollLower      = 31,
    MouthRollUpper      = 32,
    MouthShrugLower     = 33,
    MouthShrugUpper     = 34,
    MouthPressLeft      = 35,
    MouthPressRight     = 36,
    MouthLowerDownLeft  = 37,
    MouthLowerDownRight = 38,
    MouthUpperUpLeft    = 39,
    MouthUpperUpRight   = 40,
    BrowDownLeft        = 41,
    BrowDownRight       = 42,
    BrowInnerUp         = 43,
    BrowOuterUpLeft     = 44,
    BrowOuterUpRight    = 45,
    CheekPuff           = 46,
    CheekSquintLeft     = 47,
    CheekSquintRight    = 48,
    NoseSneerLeft       = 49,
    NoseSneerRight      = 50,
    TongueOut           = 51,
    Reserved52          = 52,
    Reserved53          = 53,
    Reserved54          = 54,
    Reserved55          = 55,
    Reserved56          = 56,
    Reserved57          = 57,
    Reserved58          = 58,
    Reserved59          = 59,
    Reserved60          = 60,
    COUNT               UMETA(Hidden)
};

/** Head pose + blendshapes decoded from one ARKit UDP packet */
USTRUCT(BlueprintType)
struct YOURGAME_API FARKitFaceFrame
{
    GENERATED_BODY()

    UPROPERTY(BlueprintReadOnly, Category = "ARKit")
    double Timestamp = 0.0;

    UPROPERTY(BlueprintReadOnly, Category = "ARKit")
    bool bSuccess = false;

    /** Head rotation as quaternion (X Y Z W) */
    UPROPERTY(BlueprintReadOnly, Category = "ARKit")
    FQuat HeadRotation = FQuat::Identity;

    /** Head rotation as Euler angles (Pitch Yaw Roll) in degrees */
    UPROPERTY(BlueprintReadOnly, Category = "ARKit")
    FRotator HeadEuler = FRotator::ZeroRotator;

    /** Head translation in camera space */
    UPROPERTY(BlueprintReadOnly, Category = "ARKit")
    FVector HeadTranslation = FVector::ZeroVector;

    /** 61 blendshape weights, 0–1.  Index via EARKitShape. */
    UPROPERTY(BlueprintReadOnly, Category = "ARKit")
    TArray<float> Shapes;

    FARKitFaceFrame()
    {
        Shapes.SetNumZeroed((int32)EARKitShape::COUNT);
    }
};

/** Background thread that reads the UDP socket */
class FARKitReceiverThread : public FRunnable
{
public:
    FARKitReceiverThread(class UARKitFaceReceiver* InOwner, FSocket* InSocket);

    virtual bool   Init()    override;
    virtual uint32 Run()     override;
    virtual void   Stop()    override;
    virtual void   Exit()    override;

private:
    class UARKitFaceReceiver* Owner;
    FSocket*                  Socket;
    TAtomic<bool>             bRunning{ false };
};

/**
 * UARKitFaceReceiver
 *
 * Drop onto any Actor.  Opens a UDP socket on BeginPlay and fires
 * OnFaceFrameReceived every time a valid ARKF packet arrives.
 *
 * Module dependencies (add to YourGame.Build.cs):
 *   PrivateDependencyModuleNames.AddRange(new string[] {
 *       "Networking", "Sockets"
 *   });
 */
UCLASS(ClassGroup=(Custom), meta=(BlueprintSpawnableComponent),
       DisplayName="ARKit Face Receiver")
class YOURGAME_API UARKitFaceReceiver : public UActorComponent
{
    GENERATED_BODY()

public:
    UARKitFaceReceiver();

    // ── Config ────────────────────────────────────────────────────────────

    /** Port to listen on.  Must match --arkit-port in facetracker_lite.py */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "ARKit|Config")
    int32 ListenPort = 11574;

    /**
     * Smoothing factor per frame (0 = no smoothing, 1 = frozen).
     * Applied to blendshapes on the game thread before broadcast.
     */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "ARKit|Config",
              meta = (ClampMin = "0.0", ClampMax = "1.0"))
    float SmoothingAlpha = 0.25f;

    // ── Latest data (thread-safe read from Blueprint) ─────────────────────

    UPROPERTY(BlueprintReadOnly, Category = "ARKit")
    FARKitFaceFrame LatestFrame;

    // ── Event ─────────────────────────────────────────────────────────────

    /** Fired on the game thread every time a new frame arrives */
    DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(
        FOnFaceFrameReceived, const FARKitFaceFrame&, Frame);

    UPROPERTY(BlueprintAssignable, Category = "ARKit")
    FOnFaceFrameReceived OnFaceFrameReceived;

    // ── Blueprint helpers ─────────────────────────────────────────────────

    /** Get one blendshape weight by enum index */
    UFUNCTION(BlueprintPure, Category = "ARKit")
    float GetShape(EARKitShape Shape) const;

    /** Get all 61 blendshape weights as a flat array */
    UFUNCTION(BlueprintPure, Category = "ARKit")
    TArray<float> GetAllShapes() const;

    // ── UActorComponent ───────────────────────────────────────────────────
    virtual void BeginPlay()  override;
    virtual void EndPlay(const EEndPlayReason::Type Reason) override;
    virtual void TickComponent(float DeltaTime, ELevelTick TickType,
                               FActorComponentTickFunction* ThisTickFunction) override;

    // ── Internal (called from background thread) ──────────────────────────
    void EnqueueFrame(const FARKitFaceFrame& Frame);

private:
    // Packet constants
    static constexpr int32 MAGIC_BYTES   = 4;
    static constexpr int32 ARKIT_COUNT   = 61;
    static constexpr int32 PACKET_BYTES  =
        MAGIC_BYTES          // "ARKF"
        + sizeof(double)     // timestamp
        + sizeof(uint8_t)    // success
        + sizeof(float) * 7  // quat xyzw + euler xyz
        + sizeof(float) * 3  // translation xyz
        + sizeof(float) * ARKIT_COUNT;

    FSocket*               Socket       = nullptr;
    FARKitReceiverThread*  RecvThread   = nullptr;
    FRunnableThread*       Thread       = nullptr;

    // Lock-free single-producer / single-consumer pending frame
    FCriticalSection       PendingLock;
    TOptional<FARKitFaceFrame> PendingFrame;
    bool                   bHasPending  = false;

    // Smoothed output (updated on game thread)
    FARKitFaceFrame        SmoothedFrame;

    bool TryParsePacket(const TArray<uint8>& Data, FARKitFaceFrame& OutFrame);
    void ApplySmoothing(const FARKitFaceFrame& Raw);
};
