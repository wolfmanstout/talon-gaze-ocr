#import <CoreGraphics/CoreGraphics.h>
#import <Foundation/Foundation.h>
#import <ScreenCaptureKit/ScreenCaptureKit.h>
#import <math.h>

typedef struct {
    uint8_t red;
    uint8_t green;
    uint8_t blue;
    uint8_t alpha;
} Pixel;

static void Usage(void) {
    fprintf(stderr, "Usage: mac_window_probe X Y\n");
    exit(2);
}

static NSDictionary<NSNumber *, SCWindow *> *GetShareableWindows(NSString **errorText) {
    dispatch_semaphore_t semaphore = dispatch_semaphore_create(0);
    __block SCShareableContent *shareableContent = nil;
    __block NSError *shareableError = nil;
    [SCShareableContent getShareableContentExcludingDesktopWindows:YES
                                               onScreenWindowsOnly:YES
                                                completionHandler:^(SCShareableContent *content, NSError *error) {
                                                    shareableContent = content;
                                                    shareableError = error;
                                                    dispatch_semaphore_signal(semaphore);
                                                }];
    dispatch_time_t deadline = dispatch_time(DISPATCH_TIME_NOW, 5 * NSEC_PER_SEC);
    if (dispatch_semaphore_wait(semaphore, deadline) != 0) {
        *errorText = @"timed out";
        return @{};
    }
    if (shareableError != nil || shareableContent == nil) {
        *errorText = shareableError.localizedDescription ?: @"unavailable";
        return @{};
    }

    NSMutableDictionary<NSNumber *, SCWindow *> *windows = [NSMutableDictionary dictionary];
    for (SCWindow *window in shareableContent.windows) {
        windows[@(window.windowID)] = window;
    }
    return windows;
}

static CGImageRef CaptureWindowImage(SCWindow *window, NSString **errorText) {
    SCContentFilter *filter = [[SCContentFilter alloc] initWithDesktopIndependentWindow:window];
    SCStreamConfiguration *configuration = [[SCStreamConfiguration alloc] init];
    configuration.width = MAX(1, lround(window.frame.size.width));
    configuration.height = MAX(1, lround(window.frame.size.height));
    configuration.showsCursor = NO;
    configuration.ignoreShadowsSingleWindow = YES;
    configuration.shouldBeOpaque = NO;

    dispatch_semaphore_t semaphore = dispatch_semaphore_create(0);
    __block CGImageRef image = NULL;
    __block NSError *captureError = nil;
    [SCScreenshotManager captureImageWithFilter:filter
                                  configuration:configuration
                              completionHandler:^(CGImageRef result, NSError *error) {
                                  if (result != NULL) {
                                      image = CGImageRetain(result);
                                  }
                                  captureError = error;
                                  dispatch_semaphore_signal(semaphore);
                              }];
    dispatch_time_t deadline = dispatch_time(DISPATCH_TIME_NOW, 5 * NSEC_PER_SEC);
    if (dispatch_semaphore_wait(semaphore, deadline) != 0) {
        *errorText = @"timed out";
        return NULL;
    }
    if (captureError != nil || image == NULL) {
        *errorText = captureError.localizedDescription ?: @"unavailable";
        return NULL;
    }

    return image;
}

static BOOL SamplePixel(SCWindow *window, CGPoint point, Pixel *pixel, NSString **errorText) {
    CGImageRef image = CaptureWindowImage(window, errorText);
    if (image == NULL) {
        return NO;
    }

    size_t width = CGImageGetWidth(image);
    size_t height = CGImageGetHeight(image);
    size_t bytesPerRow = width * 4;
    NSMutableData *data = [NSMutableData dataWithLength:bytesPerRow * height];
    CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();
    CGContextRef context = CGBitmapContextCreate(
        data.mutableBytes,
        width,
        height,
        8,
        bytesPerRow,
        colorSpace,
        (CGBitmapInfo)kCGImageAlphaPremultipliedLast
    );
    CGColorSpaceRelease(colorSpace);
    if (context == NULL) {
        CGImageRelease(image);
        *errorText = @"unable to allocate bitmap context";
        return NO;
    }

    CGContextTranslateCTM(context, 0, height);
    CGContextScaleCTM(context, 1, -1);
    CGContextDrawImage(context, CGRectMake(0, 0, width, height), image);
    CGContextRelease(context);
    CGImageRelease(image);

    CGRect frame = window.frame;
    NSInteger pixelX = floor((point.x - frame.origin.x) * width / frame.size.width);
    NSInteger pixelY = floor((point.y - frame.origin.y) * height / frame.size.height);
    if (pixelX < 0 || pixelX >= width || pixelY < 0 || pixelY >= height) {
        *errorText = @"point outside captured window image";
        return NO;
    }

    uint8_t *pixelBytes = (uint8_t *)data.mutableBytes
        + pixelY * bytesPerRow
        + pixelX * 4;
    pixel->red = pixelBytes[0];
    pixel->green = pixelBytes[1];
    pixel->blue = pixelBytes[2];
    pixel->alpha = pixelBytes[3];
    return YES;
}

static NSString *Quoted(NSString *value) {
    if (value == nil) {
        return @"\"\"";
    }

    NSString *escaped = [value stringByReplacingOccurrencesOfString:@"\""
                                                          withString:@"\\\""];
    return [NSString stringWithFormat:@"\"%@\"", escaped];
}

static NSString *FormatRect(CGRect rect) {
    return [NSString stringWithFormat:@"(%.0f,%.0f %.0fx%.0f)",
                                      rect.origin.x,
                                      rect.origin.y,
                                      rect.size.width,
                                      rect.size.height];
}

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        if (argc != 3) {
            Usage();
        }

        char *xEnd = NULL;
        char *yEnd = NULL;
        double x = strtod(argv[1], &xEnd);
        double y = strtod(argv[2], &yEnd);
        if (*xEnd != '\0' || *yEnd != '\0') {
            Usage();
        }

        CGPoint point = CGPointMake(x, y);
        NSArray<NSDictionary *> *windowInfos = CFBridgingRelease(
            CGWindowListCopyWindowInfo(
                kCGWindowListOptionOnScreenOnly | kCGWindowListExcludeDesktopElements,
                kCGNullWindowID
            )
        );
        if (windowInfos == nil) {
            fprintf(stderr, "Unable to read the onscreen Quartz window list.\n");
            return 1;
        }

        printf("point=(%.0f,%.0f) windows=%lu\n", point.x, point.y, (unsigned long)windowInfos.count);
        printf("order id pid layer alpha pixel owner title bounds\n");

        NSString *shareableError = nil;
        NSDictionary<NSNumber *, SCWindow *> *shareableWindows = GetShareableWindows(&shareableError);
        if (shareableError != nil) {
            printf("ScreenCaptureKit windows unavailable: %s\n", shareableError.UTF8String);
        }

        NSUInteger containingWindows = 0;
        for (NSUInteger order = 0; order < windowInfos.count; order++) {
            NSDictionary *info = windowInfos[order];
            CGRect bounds;
            NSDictionary *boundsInfo = info[(__bridge NSString *)kCGWindowBounds];
            BOOL parsedBounds = CGRectMakeWithDictionaryRepresentation(
                (__bridge CFDictionaryRef)boundsInfo,
                &bounds
            );
            if (!parsedBounds || !CGRectContainsPoint(bounds, point)) {
                continue;
            }

            containingWindows++;
            CGWindowID windowID = [info[(__bridge NSString *)kCGWindowNumber] unsignedIntValue];
            int ownerPID = [info[(__bridge NSString *)kCGWindowOwnerPID] intValue];
            int layer = [info[(__bridge NSString *)kCGWindowLayer] intValue];
            double windowAlpha = [info[(__bridge NSString *)kCGWindowAlpha] doubleValue];
            NSString *owner = info[(__bridge NSString *)kCGWindowOwnerName];
            NSString *title = info[(__bridge NSString *)kCGWindowName];

            Pixel pixel;
            NSString *pixelError = nil;
            SCWindow *shareableWindow = shareableWindows[@(windowID)];
            BOOL sampledPixel = shareableWindow != nil
                && SamplePixel(shareableWindow, point, &pixel, &pixelError);
            NSString *pixelText = sampledPixel
                ? [NSString stringWithFormat:@"rgba(%u,%u,%u,%u)", pixel.red, pixel.green, pixel.blue, pixel.alpha]
                : [NSString stringWithFormat:@"unreadable:%@", pixelError ?: @"not-shareable"];
            printf(
                "%lu %u %d %d %.3f %s %s %s %s\n",
                (unsigned long)order,
                windowID,
                ownerPID,
                layer,
                windowAlpha,
                pixelText.UTF8String,
                Quoted(owner).UTF8String,
                Quoted(title).UTF8String,
                FormatRect(bounds).UTF8String
            );
        }

        if (containingWindows == 0) {
            printf("No onscreen non-desktop Quartz windows contain the point.\n");
        }
    }

    return 0;
}
