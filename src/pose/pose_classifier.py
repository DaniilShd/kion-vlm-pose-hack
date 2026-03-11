import numpy as np

class PoseClassifier:
    def __init__(self):
        # индексы ключевых точек
        self.NOSE = 0
        self.LEFT_WRIST = 9
        self.RIGHT_WRIST = 10
        self.LEFT_ELBOW = 7
        self.RIGHT_ELBOW = 8
        self.LEFT_SHOULDER = 5
        self.RIGHT_SHOULDER = 6
        
    def check_smoking(self, keypoints, threshold=50):
        """Проверка курения: расстояние от кисти до носа"""
        nose = keypoints[self.NOSE]
        left_wrist = keypoints[self.LEFT_WRIST]
        right_wrist = keypoints[self.RIGHT_WRIST]
        
        dist_left = np.linalg.norm(nose - left_wrist)
        dist_right = np.linalg.norm(nose - right_wrist)
        
        return (dist_left < threshold) or (dist_right < threshold)
    
    def check_fighting(self, keypoints_list, threshold=100):
        """Проверка драки: руки вытянуты и близко к другому человеку"""
        if len(keypoints_list) < 2:
            return False
            
        # проверяем для каждой пары людей
        for i in range(len(keypoints_list)):
            for j in range(i+1, len(keypoints_list)):
                # расстояние между людьми (по центрам)
                center_i = np.mean(keypoints_list[i], axis=0)
                center_j = np.mean(keypoints_list[j], axis=0)
                dist = np.linalg.norm(center_i - center_j)
                
                if dist < threshold:
                    return True
        return False
    
    def check_hands_up(self, keypoints):
        """Проверка поднятых рук"""
        left_wrist = keypoints[self.LEFT_WRIST]
        right_wrist = keypoints[self.RIGHT_WRIST]
        nose = keypoints[self.NOSE]
        
        # руки выше носа?
        left_up = left_wrist[1] < nose[1]  # y координата меньше = выше
        right_up = right_wrist[1] < nose[1]
        
        return left_up or right_up
    
    def needs_vlm(self, keypoints_list, frame):
        """Решает, нужно ли отправлять кадр в VLM"""
        
        # Если нет людей - не отправляем
        if len(keypoints_list) == 0:
            return False, "no_people"
        
        reasons = []
        
        # Проверяем каждого человека
        for kps in keypoints_list:
            if self.check_smoking(kps):
                reasons.append("smoking")
            
            if self.check_hands_up(kps):
                reasons.append("hands_up")
        
        # Проверяем взаимодействие людей
        if self.check_fighting(keypoints_list):
            reasons.append("close_contact")
        
        # Отправляем в VLM если есть хоть одна причина
        if reasons:
            return True, reasons
        else:
            return False, "normal"