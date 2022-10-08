//
//  ViewController.swift
//  CoreMLProject
//
//  Created by nikolai on 10/7/22.
//

import UIKit
import CoreML
import Vision
import AVFoundation

class ViewController: UIViewController {
    
    private var bodyBox = UIView()
    private var headBox = UIView()
    private var headHitBox = UIView()
    private var stabilityBox = UIView()
    private let playerLayer = AVPlayerLayer()
    private var player: AVPlayer!
    private var output: AVPlayerItemVideoOutput?
    private var timer: Timer!
    private var latestFrame: CVPixelBuffer?
    private var rectBuffer: [CGRect] = []
    private var validHeadPosition = false
    private var currentlyStable = false

    override func viewDidLoad() {
        super.viewDidLoad()
        // Do any additional setup after loading the view.
        
        let url = Bundle.main.url(forResource: "video_2", withExtension: "mp4")
        player = AVPlayer(url: url!)
        playerLayer.player = player
        playerLayer.videoGravity = .resizeAspectFill
        self.view.layer.addSublayer(playerLayer)
        playerLayer.frame = self.view.bounds
        player.play()
        
        player.currentItem?.addObserver(self, forKeyPath:#keyPath(AVPlayerItem.status), options: [.initial, .old, .new], context:nil)
        self.timer = Timer.scheduledTimer(withTimeInterval: 0.5, repeats: true, block: { _ in
            if (self.player.timeControlStatus == .playing) {
                self.updateFrame()
                self.classifyPersion()
                self.evaluateHeadPosition()
                self.shouldStartAnalyzing()
            }
        })
        
        self.bodyBox.alpha = 0
        self.bodyBox.layer.borderColor = UIColor.systemRed.cgColor
        self.bodyBox.layer.cornerRadius = 8.0
        self.bodyBox.layer.borderWidth = 4.5
        self.playerLayer.addSublayer(self.bodyBox.layer)
        
        self.headBox.frame = CGRect(x: 0, y: 0, width: 10, height: 10)
        self.headBox.alpha = 0
        self.headBox.backgroundColor = UIColor.systemRed
        self.headBox.layer.cornerRadius = 5
        self.playerLayer.addSublayer(self.headBox.layer)
        
        self.headHitBox.frame = CGRect(x: 0, y: 0, width: 225, height: 225)
        self.headHitBox.center = self.view.center
        self.headHitBox.alpha = 0
        self.headHitBox.layer.borderColor = UIColor.systemBlue.cgColor
        self.headHitBox.layer.cornerRadius = 4.5
        self.headHitBox.layer.borderWidth = 4.5
        self.playerLayer.addSublayer(self.headHitBox.layer)
        
        self.stabilityBox.frame = CGRect(x: 0, y: 0, width: 25, height: 25)
        self.stabilityBox.center = self.view.center
        self.stabilityBox.alpha = 0
        self.stabilityBox.layer.borderColor = UIColor.systemBlue.cgColor
        self.stabilityBox.layer.cornerRadius = 4.5
        self.stabilityBox.layer.borderWidth = 4.5
        self.playerLayer.addSublayer(self.stabilityBox.layer)
    }
    
    override func observeValue(forKeyPath keyPath: String?, of object: Any?, change: [NSKeyValueChangeKey : Any]?, context: UnsafeMutableRawPointer?) {
        guard let keyPath = keyPath, let item = object as? AVPlayerItem else {
            return
        }

        switch keyPath {
        case #keyPath(AVPlayerItem.status):
            if item.status == .readyToPlay {
                self.setupOutput()
            }
        default:
            break
        }
    }
    
    
    func setupOutput() {
        guard self.output == nil else {
            return
        }
        
        let videoItem = player.currentItem!
        if videoItem.status != AVPlayerItem.Status.readyToPlay {
            return
        }
        
        let pixelBuffAttributes = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange] as [String: Any]
        
        let videoOutput = AVPlayerItemVideoOutput(pixelBufferAttributes: pixelBuffAttributes)
        videoItem.add(videoOutput)
        self.output = videoOutput
    }
    
    func updateFrame() {
        guard let output = output, let currentItem = player.currentItem else {
            return
        }

        let time = currentItem.currentTime()
        if !output.hasNewPixelBuffer(forItemTime: time) {
            return
        }
        
        guard let buffer = output.copyPixelBuffer(forItemTime: time, itemTimeForDisplay: nil) else {
            return
        }
        
        self.latestFrame = buffer
    }
    
    func classifyPersion() {
        guard let buffer = self.latestFrame else {
            return
        }
        
        let width = CVPixelBufferGetWidth(buffer)
        let height = CVPixelBufferGetHeight(buffer)
        
        let completionHandler: VNRequestCompletionHandler = { request, error in
            guard let results = request.results as? [VNHumanObservation] else {
                self.bodyBox.alpha = 0
                return
            }
            
            for result in results {
                DispatchQueue.main.async {
                    self.bodyBox.alpha = 1
                    self.bodyBox.frame = self.getConvertedRect(boundingBox: result.boundingBox, inImage: CGSize(width: width, height: height), containedIn: self.playerLayer.videoRect.size)
                    self.updateRectBuffer(frame: self.bodyBox.frame)
                }
            }
        }
        
        let personDetection = VNDetectHumanRectanglesRequest(completionHandler: completionHandler)
        personDetection.upperBodyOnly = false
        let headDetection = VNDetectHumanBodyPoseRequest { request, error in
            guard let observations = request.results as? [VNHumanBodyPoseObservation] else {
                return
            }

            // Process each observation to find the recognized body pose points.
            observations.forEach {
                self.processObservation($0)
            }
        }
        
        let requestHandler = VNSequenceRequestHandler()
        
        do {
            let model = try VNCoreMLModel(for: YOLOv3TinyFP16().model)
            try requestHandler.perform([personDetection, headDetection], on: buffer, orientation: .up)
        } catch {
            print(error.localizedDescription)
        }
    }
    
    func processObservation(_ observation: VNHumanBodyPoseObservation) {
        guard let buffer = self.latestFrame else {
            return
        }

        // Retrieve all torso points.
        guard let recognizedPoints = try? observation.recognizedPoints(.torso) else {
            return
        }

        // Torso joint names in a clockwise ordering.
        let torsoJointNames: [VNHumanBodyPoseObservation.JointName] = [
            .neck
        ]

        let width = CVPixelBufferGetWidth(buffer)
        let height = CVPixelBufferGetHeight(buffer)

        let imagePoints: [CGPoint] = torsoJointNames.compactMap {
            guard let point = recognizedPoints[$0], point.confidence > 0 else {
                return nil
            }

            return point.location
        }

        guard let point = imagePoints.first else {
            return
        }

        let center = self.getConvertedRect(boundingBox: CGRect(origin: point, size: CGSize(width: 0.1, height: 0.1)), inImage: CGSize(width: width, height: height), containedIn: self.playerLayer.videoRect.size).origin
        self.headBox.center = CGPoint(x: CGFloat(center.x), y: CGFloat(center.y + 50.0))
    }
    
    private func evaluateHeadPosition() {
        if (self.headHitBox.frame.contains(CGPoint(x: self.headBox.center.x, y: self.headBox.center.y))) {
            self.validHeadPosition = true
        } else {
            self.validHeadPosition = false
        }
    }
    
    private func updateRectBuffer(frame: CGRect) {
        self.rectBuffer.insert(frame, at: 0)
        if (rectBuffer.count > 3) {
            self.rectBuffer.popLast()
            
            var diffThreshold = 100.0
            var lastRect: CGRect = .infinite
            var lastArea = 0.0
            for rect in self.rectBuffer {
                self.currentlyStable = false
                
                if (lastRect.isInfinite) {
                    lastRect = rect
                    continue
                }
                
                self.stabilityBox.frame = rect.union(lastRect)
                lastRect = rect
                let area = self.stabilityBox.frame.size.width * self.stabilityBox.frame.size.height
                
                if (lastArea == 0.0) {
                    lastArea = area
                    continue
                }
                
                if (abs(lastArea - area) < diffThreshold) {
                    self.currentlyStable = true
                    continue
                }
                
                lastArea = area
                
            }
        }
    }
    
    private func shouldStartAnalyzing() {
        if (self.validHeadPosition && self.currentlyStable) {
            self.bodyBox.layer.borderColor = UIColor.systemGreen.cgColor
        }
    }
    
    private func getConvertedRect(boundingBox: CGRect, inImage imageSize: CGSize, containedIn containerSize: CGSize) -> CGRect {
        let rectOfImage: CGRect
        
        let imageAspect = imageSize.width / imageSize.height
        let containerAspect = containerSize.width / containerSize.height
        
        if imageAspect > containerAspect { /// image extends left and right
            let newImageWidth = containerSize.height * imageAspect /// the width of the overflowing image
            let newX = -(newImageWidth - containerSize.width) / 2
            rectOfImage = CGRect(x: newX, y: 0, width: newImageWidth, height: containerSize.height)
        } else { /// image extends top and bottom
            let newImageHeight = containerSize.width * (1 / imageAspect) /// the width of the overflowing image
            let newY = -(newImageHeight - containerSize.height) / 2
            rectOfImage = CGRect(x: 0, y: newY, width: containerSize.width, height: newImageHeight)
        }
        
        let newOriginBoundingBox = CGRect(
            x: boundingBox.origin.x,
            y: 1 - boundingBox.origin.y - boundingBox.height,
            width: boundingBox.width,
            height: boundingBox.height
        )
        
        var convertedRect = VNImageRectForNormalizedRect(newOriginBoundingBox, Int(rectOfImage.width), Int(rectOfImage.height))
        
        /// add the margins
        convertedRect.origin.x += rectOfImage.origin.x
        convertedRect.origin.y += rectOfImage.origin.y
        
        return convertedRect
    }

}

